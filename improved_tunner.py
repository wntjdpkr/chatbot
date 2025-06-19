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

# 🔧 GPU 설정
print("🔍 GPU 사용 가능 여부 확인 중...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📱 사용할 디바이스: {device}")

if torch.cuda.is_available():
    print(f"🎮 GPU 정보:")
    print(f"  - GPU 개수: {torch.cuda.device_count()}")
    print(f"  - 현재 GPU: {torch.cuda.current_device()}")
    print(f"  - GPU 이름: {torch.cuda.get_device_name()}")
    print(f"  - GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")
else:
    print("⚠️ GPU를 사용할 수 없습니다. CPU를 사용합니다.")

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

# 🎮 모델을 GPU로 이동
print(f"🚀 모델을 {device}로 이동 중...")
model = model.to(device)
print(f"✅ 모델이 {device}에 로드되었습니다.")

# ElectraClassificationHead 구조 확인
print("🔍 분류층 구조 확인:")
print(f"Classifier type: {type(model.classifier)}")
for name, module in model.classifier.named_children():
    print(f"  - {name}: {module}")

# 4. 모델 확장: 44개 → 45개 라벨
print("🔧 모델 출력층 확장 중...")

# ElectraClassificationHead에서 실제 출력층 접근
# ElectraClassificationHead 구조: dense -> dropout -> out_proj
old_classifier = model.classifier
if hasattr(old_classifier, 'out_proj'):
    # out_proj가 실제 분류를 담당하는 레이어
    old_out_proj = old_classifier.out_proj
    old_weight = old_out_proj.weight.data  # [44, hidden_size]
    old_bias = old_out_proj.bias.data  # [44]
    hidden_size = old_weight.shape[1]
    print(f"기존 out_proj 크기: {old_weight.shape}")
elif hasattr(old_classifier, 'dense') and hasattr(old_classifier, 'out_proj'):
    # 일부 버전에서는 dense -> out_proj 구조
    old_out_proj = old_classifier.out_proj
    old_weight = old_out_proj.weight.data
    old_bias = old_out_proj.bias.data
    hidden_size = old_weight.shape[1]
else:
    # 다른 구조일 경우 직접 확인
    print("❌ 예상치 못한 분류층 구조입니다.")
    print("분류층 내부 구조:")
    for name, param in old_classifier.named_parameters():
        print(f"  {name}: {param.shape}")

    # 마지막 레이어를 찾아서 사용
    last_layer = None
    for name, module in old_classifier.named_modules():
        if isinstance(module, nn.Linear):
            last_layer = module

    if last_layer is not None:
        old_weight = last_layer.weight.data
        old_bias = last_layer.bias.data
        hidden_size = old_weight.shape[1]
        print(f"✅ 마지막 Linear 레이어 발견: {old_weight.shape}")
    else:
        raise ValueError("분류층에서 Linear 레이어를 찾을 수 없습니다.")

# 새로운 출력층 생성 (45개 라벨) - GPU에서 생성
print(f"🔧 새로운 출력층을 {device}에서 생성 중...")
new_out_proj = nn.Linear(hidden_size, len(labels)).to(device)

# 기존 44개 라벨의 가중치 복사
with torch.no_grad():
    new_out_proj.weight.data[:44] = old_weight
    new_out_proj.bias.data[:44] = old_bias

    # 새로운 라벨(허무)의 가중치 초기화
    # 유사한 감정들(슬픔, 절망, 허무)의 평균으로 초기화
    similar_emotions = [5, 19, 25, 36]  # 슬픔, 절망, 패배/자기혐오, 서러움
    similar_weights = old_weight[similar_emotions].mean(dim=0)
    similar_bias = old_bias[similar_emotions].mean()

    boost_factor = 2.0  # 예시값, 2.0까지도 실험 가능
    new_out_proj.weight.data[44] = similar_weights * boost_factor
    new_out_proj.bias.data[44] = similar_bias * boost_factor

# ElectraClassificationHead의 out_proj만 교체
if hasattr(old_classifier, 'out_proj'):
    old_classifier.out_proj = new_out_proj
else:
    # 전체 분류층을 새로 만들어야 하는 경우
    from transformers.models.electra.modeling_electra import ElectraClassificationHead

    # 새로운 ElectraClassificationHead 생성
    config = model.config
    config.num_labels = len(labels)

    new_classifier = ElectraClassificationHead(config).to(device)
    # 기존 dense 레이어는 유지하고 out_proj만 교체
    if hasattr(old_classifier, 'dense'):
        new_classifier.dense = old_classifier.dense
    if hasattr(old_classifier, 'dropout'):
        new_classifier.dropout = old_classifier.dropout
    new_classifier.out_proj = new_out_proj

    model.classifier = new_classifier

model.config.num_labels = len(labels)
model.config.label2id = label2id
model.config.id2label = id2label
model.config.problem_type = "single_label_classification"

print(f"✅ 모델 확장 완료: {model.config.num_labels}개 라벨")
print(f"🎮 모델이 {device}에서 실행 중")

# 5. 데이터 구조 확인 및 전처리
print("🔍 데이터셋 구조 확인:")
print(f"데이터셋 컬럼: {dataset.column_names}")
print(f"데이터셋 타입: {type(dataset)}")
print(f"데이터셋 길이: {len(dataset)}")

# 첫 번째 샘플 안전하게 확인
try:
    first_sample = dataset[0]
    print(f"첫 번째 샘플: {first_sample}")
    print(f"첫 번째 샘플 타입: {type(first_sample)}")
except Exception as e:
    print(f"첫 번째 샘플 접근 오류: {e}")


# 전처리 함수 개선
def preprocess_function(examples):
    """
    배치 단위로 토크나이징과 라벨 인코딩을 동시에 처리
    """
    try:
        # 입력 확인
        sentences = examples["sentence"]
        labels = examples["label"]

        print(f"전처리 중 - 배치 크기: {len(sentences)}")
        print(f"첫 번째 문장: {sentences[0] if sentences else 'None'}")
        print(f"첫 번째 라벨: {labels[0] if labels else 'None'}")

        # 토크나이징
        tokenized = tokenizer(
            sentences,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors=None  # 리스트로 반환
        )

        # 라벨 인코딩
        encoded_labels = []
        for label in labels:
            if label in label2id:
                encoded_labels.append(label2id[label])
            else:
                print(f"❌ 알 수 없는 라벨: {label}")
                encoded_labels.append(44)  # 기본값으로 허무 설정

        tokenized["labels"] = encoded_labels

        print(f"전처리 완료 - 라벨들: {encoded_labels[:3]}...")  # 첫 3개만 출력

        return tokenized

    except Exception as e:
        print(f"❌ 전처리 중 오류 발생: {e}")
        print(f"Examples 타입: {type(examples)}")
        print(f"Examples 내용: {examples}")
        raise e


# 데이터 전처리 적용
print("🔄 데이터 전처리 중...")
try:
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=8,  # 작은 배치로 시작
        remove_columns=dataset.column_names  # 원본 컬럼 제거
    )
    print("✅ 전처리 완료!")
except Exception as e:
    print(f"❌ 전처리 실패: {e}")
    print("개별 처리로 시도...")

    # 개별 처리 fallback
    processed_data = []
    for i, sample in enumerate(dataset):
        try:
            sentence = sample["sentence"]
            label = sample["label"]

            # 토크나이징
            tokenized = tokenizer(
                sentence,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors=None
            )

            # 라벨 인코딩
            tokenized["labels"] = label2id[label]
            processed_data.append(tokenized)

            if i < 3:  # 첫 3개만 로그
                print(f"개별 처리 완료 {i}: 라벨 {label} -> {tokenized['labels']}")

        except Exception as inner_e:
            print(f"샘플 {i} 처리 실패: {inner_e}")

    # 새 데이터셋 생성
    from datasets import Dataset

    dataset = Dataset.from_list(processed_data)
    print("✅ 개별 처리로 전처리 완료!")

print(f"전처리 후 컬럼: {dataset.column_names}")
if len(dataset) > 0:
    print(f"전처리 후 첫 번째 샘플 키: {list(dataset[0].keys())}")

# 데이터 타입 확인
sample = dataset[0]
print("📊 데이터 타입 확인:")
for key, value in sample.items():
    print(
        f"  {key}: {type(value)} - {value if key == 'labels' else f'길이 {len(value) if isinstance(value, list) else value}'}")

# 몇 개 샘플의 라벨 확인
print("\n🎯 라벨 분포 확인:")
# 안전한 방식으로 라벨 접근
try:
    # 방법 1: 개별 샘플 접근
    labels_list = []
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        labels_list.append(sample['labels'])
    print(f"첫 10개 샘플의 라벨: {labels_list}")
except Exception as e:
    print(f"개별 접근 실패: {e}")

    # 방법 2: 컬럼 직접 접근
    try:
        labels_list = dataset['labels'][:10]
        print(f"첫 10개 샘플의 라벨: {labels_list}")
    except Exception as e2:
        print(f"컬럼 접근도 실패: {e2}")

        # 방법 3: 전체 데이터셋 정보 출력
        print("데이터셋 전체 정보:")
        print(f"데이터셋 타입: {type(dataset)}")
        print(f"데이터셋 길이: {len(dataset)}")
        print(f"첫 번째 샘플 타입: {type(dataset[0])}")
        if hasattr(dataset, 'features'):
            print(f"Features: {dataset.features}")

# 모든 라벨이 44(허무)인지 확인
try:
    # 방법 1: 컬럼 직접 접근
    if hasattr(dataset, 'features') and 'labels' in dataset.features:
        unique_labels = set(dataset['labels'])
        labels_count_44 = sum(1 for label in dataset['labels'] if label == 44)
    else:
        # 방법 2: 개별 접근
        all_labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            all_labels.append(sample['labels'])
        unique_labels = set(all_labels)
        labels_count_44 = sum(1 for label in all_labels if label == 44)

    print(f"데이터셋의 고유 라벨: {unique_labels}")
    print(f"라벨 44('허무')가 {labels_count_44}개 있습니다.")
except Exception as e:
    print(f"라벨 확인 중 오류: {e}")
    print("개별 샘플 3개 확인:")
    for i in range(min(3, len(dataset))):
        print(f"  샘플 {i}: {dataset[i]}")

# 데이터셋 길이 확인
print(f"총 학습 샘플 수: {len(dataset)}")

# 토크나이징 결과 확인
try:
    first_sample = dataset[0]
    print(f"\nInput IDs 길이 확인: {len(first_sample['input_ids'])}")
    print(f"Attention mask 길이 확인: {len(first_sample['attention_mask'])}")
    print(f"Labels 타입: {type(first_sample['labels'])}, 값: {first_sample['labels']}")
except Exception as e:
    print(f"토크나이징 결과 확인 실패: {e}")

# 데이터셋 형태 최종 점검
print(f"\n데이터셋 features: {dataset.features}")

# ✅ 44번 (허무)만 학습하도록 파라미터 선택
def get_trainable_params_only_void():
    target_params = []
    for name, param in model.named_parameters():
        if "classifier.out_proj.weight" in name:
            # 44번째 weight만 학습
            sliced = torch.nn.Parameter(param.data[44:45])
            target_params.append({"params": sliced})
        elif "classifier.out_proj.bias" in name:
            # 44번째 bias만 학습
            sliced = torch.nn.Parameter(param.data[44:45])
            target_params.append({"params": sliced})
    return target_params
# 6. Few-shot Learning을 위한 커스텀 트레이너
class FewShotTrainer(Trainer):
    def __init__(self, *args, old_model_state=None, regularization_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.old_model_state = old_model_state
        self.regularization_weight = regularization_weight
        # GPU device 정보 저장
        self.device = next(self.model.parameters()).device
        print(f"🎮 Trainer가 {self.device}에서 실행됩니다.")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        파멸적 망각 방지를 위한 정규화된 손실 함수
        모델의 기본 loss 계산을 우회하여 직접 계산
        """
        # 🎮 입력 데이터가 올바른 디바이스에 있는지 확인
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(model.device)

        # num_items_in_batch 파라미터 처리 (최신 버전 대응)
        num_items_in_batch = kwargs.get('num_items_in_batch', None)

        # 라벨 추출
        labels = inputs.get("labels")

        # 모델의 기본 loss 계산을 우회하기 위해 labels 제거
        inputs_no_labels = {k: v for k, v in inputs.items() if k != "labels"}

        # 모델 forward (loss 계산 없이)
        outputs = model(**inputs_no_labels)
        logits = outputs.logits

        # 직접 loss 계산
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # 올바른 차원 사용: logits의 마지막 차원 (45)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            loss = None

        # outputs 객체에 loss 추가
        outputs.loss = loss

        # 기존 가중치와의 차이에 대한 정규화 항 추가 (파멸적 망각 방지)
        if self.old_model_state is not None and self.regularization_weight > 0 and loss is not None:
            reg_loss = 0
            reg_count = 0

            for name, param in model.named_parameters():
                if name in self.old_model_state and 'classifier.out_proj.weight' in name:
                    # 기존 44개 라벨의 가중치만 정규화
                    old_weight = self.old_model_state[name][:44].to(param.device)  # GPU로 이동
                    new_weight = param[:44]
                    reg_loss += F.mse_loss(new_weight, old_weight)
                    reg_count += 1

            if reg_count > 0:
                total_reg_loss = self.regularization_weight * reg_loss
                loss += total_reg_loss
                outputs.loss = loss  # outputs 업데이트

                # 로깅 (가끔씩만)
                if hasattr(self, '_step_count'):
                    self._step_count += 1
                else:
                    self._step_count = 1

                if self._step_count % 50 == 0:  # 50스텝마다 로깅
                    print(f"📊 Step {self._step_count}: Loss={loss:.4f}, Reg_loss={total_reg_loss:.4f}")

        return (loss, outputs) if return_outputs else loss


# 7. 기존 모델 상태 저장 (정규화용) - 새로 생성된 모델 기준
print("💾 새 모델의 기본 상태 저장 (정규화용)...")
old_model_state = {}
for name, param in model.named_parameters():
    old_model_state[name] = param.data.clone().cpu()  # CPU에 저장하여 메모리 절약

print(f"✅ {len(old_model_state)}개 파라미터 상태 저장 완료")

# 8. 학습 설정 (Few-shot Learning에 최적화) - GPU 사용 고려
training_args = TrainingArguments(
    output_dir="./results_void",
    evaluation_strategy="no",
    logging_steps=5,
    learning_rate=1e-5,  # 낮은 학습률로 안정적 학습
    per_device_train_batch_size=4,  # GPU 메모리에 따라 조정 가능
    num_train_epochs=10,  # 더 많은 에포크
    weight_decay=0.01,
    logging_dir="./logs",
    save_steps=50,
    warmup_steps=10,  # Warmup 추가
    gradient_accumulation_steps=2,  # 그래디언트 누적
    dataloader_drop_last=False,
    remove_unused_columns=True,
    dataloader_pin_memory=True if torch.cuda.is_available() else False,  # GPU 사용시 메모리 핀
    ignore_data_skip=True,  # 데이터 스킵 오류 방지
    ddp_find_unused_parameters=False,  # DDP 관련 안정성
    report_to=[],  # wandb 등 외부 로깅 비활성화
    fp16=False,  # GPU 사용 시 mixed precision 활성화
    torch_compile=False,  # 호환성을 위해 비활성화
)

print(f"🎮 학습 설정:")
print(f"  - 디바이스: {device}")
print(f"  - Mixed Precision (FP16): {training_args.fp16}")
print(f"  - 배치 크기: {training_args.per_device_train_batch_size}")
print(f"  - 그래디언트 누적: {training_args.gradient_accumulation_steps}")

# 9. Few-shot 트레이너 설정
trainer = FewShotTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    old_model_state=old_model_state,
    regularization_weight=0.1,
    optimizers=(torch.optim.AdamW(get_trainable_params_only_void(), lr=1e-5), None)
)

# 10. 학습 전 모델 상태 출력
print("\n🔍 학습 전 최종 모델 정보:")
print(f"🔢 출력층: {model.classifier}")
print(f"🎯 Config의 라벨 수: {model.config.num_labels}")
print(f"🆔 Model의 num_labels: {getattr(model, 'num_labels', 'None')}")
print(f"🎮 모델 디바이스: {next(model.parameters()).device}")
if hasattr(model.classifier, 'out_proj'):
    print(f"📊 새 라벨 초기 가중치 norm: {model.classifier.out_proj.weight.data[44].norm():.4f}")

# GPU 메모리 사용량 확인 (GPU 사용 시)
if torch.cuda.is_available():
    print(f"🔋 GPU 메모리 사용량:")
    print(f"  - 할당된 메모리: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB")
    print(f"  - 캐시된 메모리: {torch.cuda.memory_reserved() / 1024 ** 3:.2f}GB")

# 11. 파인튜닝 시작
print("\n🚀 Few-shot Learning 기반 파인튜닝 시작...")
trainer.train()

# 12. 학습 후 검증
print("\n✅ 학습 완료! 모델 검증 중...")

# 간단한 테스트 - GPU 사용 고려
test_sentences = [
    "모든 게 무의미하게 느껴진다.",  # 허무
    "정말 기쁘고 행복하다.",  # 기쁨
    "너무 화가 난다.",  # 화남
    "삶에 의미가 있을까 싶다."  # 허무
]

model.eval()
with torch.no_grad():
    for sentence in test_sentences:
        # 🎮 입력을 올바른 디바이스로 이동
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # GPU로 이동

        outputs = model(**inputs)
        predicted_id = torch.argmax(outputs.logits, dim=-1).item()
        confidence = torch.softmax(outputs.logits, dim=-1).max().item()

        print(f"📝 문장: '{sentence}'")
        print(f"🎯 예측: {labels[predicted_id]} (신뢰도: {confidence:.3f})")
        print()

# GPU 메모리 정리
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("🧹 GPU 메모리 캐시 정리 완료")

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
print(f"- 모델 아키텍처: {type(model).__name__}")
print(f"- Config num_labels: {model.config.num_labels}")
print(f"- 실행 디바이스: {device}")

# 최종 GPU 메모리 사용량 (GPU 사용 시)
if torch.cuda.is_available():
    print(f"\n🔋 최종 GPU 메모리 사용량:")
    print(f"  - 할당된 메모리: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB")
    print(f"  - 최대 사용 메모리: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")

print("\n🎉 GPU 최적화된 Few-shot Learning 기반 감정 분류 모델 확장 완료!")
