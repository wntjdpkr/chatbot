from transformers import AutoTokenizer, ElectraForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# 1. 라벨 목록 정의 (기존 44개 + 허무 추가)
labels = {
    0: '불평/불만', 1: '환영/호의', 2: '감동/감탄', 3: '지긋지긋', 4: '고마움', 5: '슬픔', 6: '화남/분노',
    7: '존경', 8: '기대감', 9: '우쭐댐/무시함', 10: '안타까움/실망', 11: '비장함', 12: '의심/불신',
    13: '뿌듯함', 14: '편안/쾌적', 15: '신기함/관심', 16: '아껴주는', 17: '부끄러움', 18: '공포/무서움',
    19: '절망', 20: '한심함', 21: '역겨움/징그러움', 22: '짜증', 23: '어이없음', 24: '없음', 25: '패배/자기혐오',
    26: '귀찮음', 27: '힘듦/지침', 28: '즐거움/신남', 29: '깨달음', 30: '죄책감', 31: '증오/혐오',
    32: '흐뭇함(귀여움/예쁨)', 33: '당황/난처', 34: '경악', 35: '부담/안_내킴', 36: '서러움',
    37: '재미없음', 38: '불쌍함/연민', 39: '놀람', 40: '행복', 41: '불안/걱정', 42: '기쁨',
    43: '안심/신뢰', 44: '허무'
}
label2id = {v: k for k, v in labels.items()}
id2label = labels

print(f"📊 총 라벨 수: {len(labels)}개")
print(f"🆕 새로 추가된 라벨: {labels[44]} (ID: 44)")

# 2. 데이터 로딩
with open("finetune.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"📄 새 라벨 학습 데이터: {len(raw_data)}개")
dataset = Dataset.from_list(raw_data)

# 3. 토크나이저 및 모델 로딩
model_name = "searle-j/kote_for_easygoing_people"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 기존 모델 로드 (44개 라벨)
print("🔄 기존 모델 로딩 중...")
model = ElectraForSequenceClassification.from_pretrained(model_name)
print(f"기존 모델 라벨 수: {model.config.num_labels}")

# 4. 모델 확장: 44개 → 45개 라벨
print("🔧 모델 출력층 확장 중...")

# 기존 가중치 보존
old_classifier = model.classifier.out_proj
old_weight = old_classifier.weight.data
old_bias = old_classifier.bias.data

# 새로운 분류층 생성 (45개 라벨)
new_classifier = nn.Linear(model.config.hidden_size, len(labels))

# 기존 44개 라벨의 가중치 복사
with torch.no_grad():
    new_classifier.weight.data[:44] = old_weight
    new_classifier.bias.data[:44] = old_bias

    # 새로운 라벨(허무)의 가중치 초기화
    # 유사한 감정들(슬픔, 절망, 허무)의 평균으로 초기화
    similar_emotions = [5, 19, 25, 36]  # 슬픔, 절망, 패배/자기혐오, 서러움
    similar_weights = old_weight[similar_emotions].mean(dim=0)
    similar_bias = old_bias[similar_emotions].mean()

    new_classifier.weight.data[44] = similar_weights
    new_classifier.bias.data[44] = similar_bias

# 모델 업데이트
model.classifier.out_proj = new_classifier
model.config.num_labels = len(labels)
model.config.label2id = label2id
model.config.id2label = id2label
model.config.problem_type = "single_label_classification"

print(f"✅ 모델 확장 완료: {model.config.num_labels}개 라벨")


# 5. 전처리 함수
def preprocess(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)


def encode_label(example):
    example["label"] = label2id[example["label"]]
    return example


dataset = dataset.map(preprocess, batched=True)
dataset = dataset.map(encode_label)


# 6. Few-shot Learning을 위한 커스텀 트레이너
class FewShotTrainer(Trainer):
    def __init__(self, *args, old_model_state=None, regularization_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.old_model_state = old_model_state
        self.regularization_weight = regularization_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        파멸적 망각 방지를 위한 정규화된 손실 함수
        """
        # 기본 분류 손실
        outputs = model(**inputs)
        loss = outputs.loss

        # 기존 가중치와의 차이에 대한 정규화 항 추가
        if self.old_model_state is not None:
            reg_loss = 0
            for name, param in model.named_parameters():
                if name in self.old_model_state and 'classifier.weight' in name:
                    # 기존 44개 라벨의 가중치만 정규화
                    old_weight = self.old_model_state[name][:44]
                    new_weight = param[:44]
                    reg_loss += F.mse_loss(new_weight, old_weight)

            loss += self.regularization_weight * reg_loss

        return (loss, outputs) if return_outputs else loss


# 7. 기존 모델 상태 저장 (정규화용)
old_model_state = {}
for name, param in model.named_parameters():
    old_model_state[name] = param.data.clone()

# 8. 학습 설정 (Few-shot Learning에 최적화)
training_args = TrainingArguments(
    output_dir="./results_void",
    evaluation_strategy="no",
    logging_steps=5,
    learning_rate=1e-5,  # 낮은 학습률로 안정적 학습
    per_device_train_batch_size=4,  # 작은 배치 크기
    num_train_epochs=10,  # 더 많은 에포크
    weight_decay=0.01,
    logging_dir="./logs",
    save_steps=50,
    warmup_steps=10,  # Warmup 추가
    gradient_accumulation_steps=2,  # 그래디언트 누적
    dataloader_drop_last=False,
    remove_unused_columns=False,
)

# 9. Few-shot 트레이너 설정
trainer = FewShotTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    old_model_state=old_model_state,
    regularization_weight=0.1  # 정규화 가중치
)

# 10. 학습 전 모델 상태 출력
print("\n🔍 학습 전 모델 정보:")
print(f"🔢 출력층: {model.classifier}")
print(f"🎯 Config의 라벨 수: {model.config.num_labels}")
print(f"📊 새 라벨 초기 가중치 norm: {model.classifier.out_proj.weight.data[44].norm():.4f}")

# 11. 파인튜닝 시작
print("\n🚀 Few-shot Learning 기반 파인튜닝 시작...")
trainer.train()

# 12. 학습 후 검증
print("\n✅ 학습 완료! 모델 검증 중...")

# 간단한 테스트
test_sentences = [
    "모든 게 무의미하게 느껴진다.",  # 허무
    "정말 기쁘고 행복하다.",  # 기쁨
    "너무 화가 난다.",  # 화남
    "삶에 의미가 있을까 싶다."  # 허무
]

model.eval()
with torch.no_grad():
    for sentence in test_sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        predicted_id = torch.argmax(outputs.logits, dim=-1).item()
        confidence = torch.softmax(outputs.logits, dim=-1).max().item()

        print(f"📝 문장: '{sentence}'")
        print(f"🎯 예측: {labels[predicted_id]} (신뢰도: {confidence:.3f})")
        print()

# 13. 모델 저장
save_path = "./custom_kote_with_void"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"💾 파인튜닝 완료! 모델이 '{save_path}'에 저장되었습니다.")
print(f"🏷️ 총 {len(labels)}개 감정 분류 가능")
print(f"🆕 새로 추가된 감정: '{labels[44]}'")

# 14. 최종 모델 정보 출력
print("\n📋 최종 모델 정보:")
print(f"- 기존 라벨: 0~43 (44개)")
print(f"- 새 라벨: 44 (허무)")
print(f"- 총 라벨 수: {len(labels)}개")
print(f"- 학습 데이터: {len(raw_data)}개 (허무 감정)")
print(f"- 저장 위치: {save_path}")
