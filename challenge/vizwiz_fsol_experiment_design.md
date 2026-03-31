# VizWiz FSOL Challenge — Experiment Design Document

---

## 1. Task Definition

### 1.1 Challenge Structure

```
Training data:   VizWiz-FewShot (base) — 100 non-private categories, 4,229 images, 8,043 instances
Support data:    BIV-Priv-Seg — 16 private categories, 16 images (1-shot, one per category)
Query data:      BIV-Priv-Seg — 1,056 images (labels unavailable)
Dev data:        BIV-Priv-Seg — 158 images (for local validation)

Constraint:      No external training data allowed
Evaluation:      mAP, AP50 (COCO-style) for both bbox and segmentation
Submission:      COCO-format JSON with image_id, category_id, bbox, segmentation, score
```

### 1.2 16 Private Novel Categories

```
bank statement, bill or receipt, business card, condom box, condom packet,
credit or debit card, doctor's prescription, pill bottle, letter with address,
local newspaper, medical record document, mortgage or investment report,
pregnancy test, pregnancy test box, tattoo sleeve, transcript
```

### 1.3 Task의 본질: 단순 few-shot이 아님

이 challenge는 base/novel을 non-private/private으로 의도적으로 나누었다.
단순히 "새 category를 1장 보고 찾는 것"이 아니라,
**non-private만 학습한 모델이 1-shot private reference로 region-level privacy discrimination을 수행할 수 있는가**가 진짜 질문이다.

다만 BIV-Priv-Seg 논문에서 이미 확인된 사실:
- Prompt I (category-specific): AP50 60.2%
- Prompt II ("private object"): AP50 18.8%
- Prompt III (category 나열 + "private"): AP50 9.4%

**"Privacy"라는 추상 개념을 직접 주입하면 성능이 박살난다.**
따라서 실험 설계에서는 **few-shot localization + domain gap (non-private → private)** 관점으로 접근하되,
privacy-aware framing은 분석과 서술에서만 활용한다.

---

## 2. Baseline 분석: LLM2Seg (2025 챌린지 1등)

### 2.1 Pipeline

```
[Fine-tune]
Support 16장 → augmentation (shift, crop, rotate, brightness) → Grounding DINO fine-tune

[Inference]
Query image → Real-ESRGAN (super-resolution)
           → GPT-4.1 (category inference, support 안 봄)
           → text label + query image → fine-tuned G-DINO → bbox
           → SAM → segmentation mask
```

### 2.2 Results

| Setting | bbox mAP | bbox AP50 | seg mAP | seg AP50 |
|---------|----------|-----------|---------|----------|
| No preprocess, No fine-tune | 51.63 | 54.78 | 51.46 | 54.48 |
| No preprocess, Fine-tune | 58.91 | 61.52 | 57.68 | 60.04 |
| Preprocess + Fine-tune | **61.63** | **64.95** | **60.14** | **63.49** |

### 2.3 핵심 약점

1. **Support의 visual information이 inference에 직접 관여하지 않음.**
   GPT-4.1은 support image를 보지 않고 사전지식만으로 category 추론.
   Few-shot이라는 조건을 open-vocab으로 우회함.

2. **VizWiz-FewShot base 100개 category를 학습에 미활용.**
   공식 허용된 4,229장 8,043 instance를 버리고, support 16장만 augmentation해서 fine-tune.

3. **Augmentation 디테일 미공개.**
   파라미터, 총 데이터 수, augmentation ratio 등 재현 불가.

4. **Reference [4] 오인용.**
   Real-ESRGAN 논문을 fine-tune 방법론 인용으로 사용.

### 2.4 BIV-Priv-Seg 원논문 benchmark과의 비교

| Method | k-shot | mAP | AP50 |
|--------|--------|-----|------|
| DeFRCN (전통 few-shot) | 1 | 12.7 | 21.7 |
| YOLACT (전통 few-shot) | 1 | 18.1 | 24.7 |
| GroundingDINO+SAM (zero-shot, Prompt I) | 0 | — | 60.2 |
| GLaMM (zero-shot, Prompt I) | 0 | — | 44.7 |
| **LLM2Seg** | **1** | **61.63** | **64.95** |

**핵심 시사점:**
- 전통 few-shot 방법 (DeFRCN, YOLACT)은 1-shot에서 처참한 성능 (mAP ~15%)
- VLM 기반 접근이 5배 이상 우수
- 텍스트 유무가 결정적 변수: 텍스트 있는 object에서 성능 2배+
- Private object의 74.5%가 텍스트 포함 → VLM의 텍스트 이해 능력이 핵심

---

## 3. 우리 접근: Two-Track Design

### 3.1 설계 원칙

```
1. VizWiz-FewShot base를 적극 활용 (LLM2Seg 대비 핵심 차별점)
2. Support의 visual information을 inference에 직접 관여시킴
3. 동일 모델 스택에서 시작하여 개선이 설계에서 온 것임을 증명
4. 두 track의 결과를 같은 evaluation split에서 비교
```

### 3.2 Track A: Grounding-centric Hybrid

```
Trainable core:  Grounding DINO
Training data:   VizWiz-FewShot base + augmented support
VLM role:        Qwen3-VL-8B — category/reference understanding, text reasoning 보조
Segmentation:    SAM (학습 없이 후처리)
목표:            챌린지 숫자 극대화
```

**Track A에서의 VLM 활용:**
- Support image를 보고 detailed description 생성 → G-DINO의 text prompt로 활용
- Category inference 시 support image를 직접 참조 (LLM2Seg과 차별화)
- 텍스트 이해 능력으로 private object의 핵심 특성 (문서, 카드 등) 활용

### 3.3 Track B: VLM-centric Single Model

```
Trainable core:  Qwen3-VL-8B (LoRA)
Training data:   VizWiz-FewShot base → support-query pair format으로 변환
Support role:    Inference 시 직접 조건으로 입력 (in-context)
Segmentation:    SAM (후처리)
목표:            정량 결과 확보 → Paper 1 + 과제 성과
```

**Track B의 핵심 구조:**
```
Training:
  Base category에서 이미지 2장을 뽑아
  하나는 support (reference), 하나는 query로 구성
  "이 support와 같은 물체를 query에서 찾아라" → bbox 출력

Inference:
  Private support 1장 + query image → Qwen3-VL → bbox
```

이 구조에서 모델이 학습하는 것은 **특정 category가 아니라 "reference 보고 같은 걸 찾는 능력"** 자체.
Category 지식이 아닌 matching 능력을 학습하므로, novel private category로의 transfer가 자연스러움.

### 3.4 Base Model 선정: Qwen3-VL-8B

| 항목 | Qwen3-VL-8B |
|------|-------------|
| Native grounding | O (bbox를 텍스트 토큰으로 직접 출력) |
| Multi-image input | O (support + query 동시 입력 가능) |
| Thinking mode | O (reasoning chain 자동 생성 → Paper 1에 활용) |
| Fine-tune | LoRA, A100 48GB에서 가능 |
| Context window | 256K (1M 확장 가능) |
| 제안서 정합성 | "오픈소스 LVLM, RefCOCO format, 위치 좌표를 언어 토큰으로" |

**LLM2Seg baseline 재현 시:** GPT-4.1 → Qwen3-VL-8B로 대체.
동일 Grounding DINO + SAM 스택에서 시작하여, 개선이 모델이 아닌 설계에서 온 것을 증명.

### 3.5 결과 활용 매핑

```
Track A 결과 → 챌린지 제출 (숫자 극대화)
Track B 결과 → Paper 1 실험 데이터 + 과제 1차년도 성과
A vs B 비교 → Extended abstract 분석
             → "모듈 파이프라인 vs 단일 VLM" 비교
             → Paper 1: "단일 VLM의 한계와 가능성"

어떤 결과가 나와도 버릴 게 없는 구조:
  A > B  → 챌린지 제출은 A, Paper 1에서 "단일 VLM 한계 분석"
  A ≈ B  → 단일 VLM이 모듈 수준임을 증명, 제안서에 강력한 근거
  A > B지만 B도 의미있는 수준 → 둘 다 제출 + 분석
```

---

## 4. Evaluation: 3-Fold Pseudo-Novel Simulation

### 4.1 왜 필요한가

- Eval server가 현재 maintenance 중
- Query set에 GT가 없어 로컬 평가 불가
- Method 개발과 hyperparameter tuning을 위한 validation 필요

### 4.2 설계 원칙

```
1. Person(39)은 모든 fold에서 항상 train (736장, co-occurrence 지배)
2. 강한 co-occurrence pair는 같은 fold (keyboard+monitor, dog+dog_collar 등)
3. Private-like category를 각 fold에 분배하여 실제 task 시뮬레이션
4. Pseudo-novel category 포함 이미지는 해당 fold의 train에서 제외 (데이터 오염 방지)
```

### 4.3 데이터 특성

```
전체: 4,229 images, 8,043 instances (이미지당 평균 ~1.9 objects)
Multi-category images: 1,856장 (44%)
Single-category images: 2,373장 (56%)

Top co-occurrence pairs:
  keyboard + monitor:     79 images
  person + rug:           74 images
  person + shoe:          56 images
  cup + person:           49 images
  dog + dog_collar:       39 images
  pillow + bed:           36 images
  oven + dial:            36 images
```

### 4.4 Private-like Base Category 매핑

실제 novel 16개 private category와 시각적/의미적으로 유사한 base category:

```
[문서류] → bank statement, letter, prescription, transcript 등과 유사
  newspaper(63,16img), envelope(74,63img), magazine(76,24img),
  receipt(90,19img), food_menu(75,20img), gift_card(73,17img)

[용기류] → pill bottle, condom 등과 유사
  bottle(33,226img), tube(86,62img), packet(59,87img), cereal_box(85,37img)

[소형물체] → credit card, business card 등과 유사
  wallet(52,33img), cash(80,71img), coin(66,32img), calculator(89,25img)

[전자기기] → 형태적 유사성
  cell_phone(84,83img), ipad(100,22img)
```

### 4.5 3-Fold Split 구성

```json
{
  "split1": {
    "private_like": [63, 33, 52, 84, 74, 86],
    "pseudo_novel": [63, 33, 52, 84, 74, 86, 6, 9, 3, 7, 1, 21, 38, 54,
                     47, 48, 13, 44, 34, 32, 94, 51, 5, 50, 69, 80, 81,
                     67, 57, 41, 42, 49, 25],
    "n_pseudo_novel": 33
  },
  "split2": {
    "private_like": [75, 73, 76, 59, 66],
    "pseudo_novel": [75, 73, 76, 59, 66, 29, 18, 8, 43, 14, 24, 17, 46,
                     45, 37, 92, 30, 31, 58, 12, 22, 27, 28, 70, 35, 19,
                     77, 83, 61, 65, 79, 78, 88],
    "n_pseudo_novel": 33
  },
  "split3": {
    "private_like": [90, 85, 89, 100, 23, 98],
    "pseudo_novel": [90, 85, 89, 100, 23, 98, 2, 4, 10, 11, 15, 16, 20,
                     26, 36, 40, 53, 55, 56, 60, 62, 64, 68, 71, 72, 82,
                     87, 91, 93, 95, 96, 97, 99],
    "n_pseudo_novel": 33
  }
}
```

### 4.6 Split별 데이터 통계

```
             pseudo-novel   train cats    contaminated   usable       usable annotations
             categories     (+ person)    images         train imgs   (/ total)
Split 1:     33             66 + person   2,112          2,117        3,986 / 5,303 (75.2%)
Split 2:     33             66 + person   1,877          2,352        3,600 / 5,215 (69.0%)
Split 3:     33             66 + person   1,228          3,001        5,038 / 6,363 (79.2%)
```

Split 간 불균형이 있으나 (split 3이 데이터 많음), method 개발용 validation으로는 충분.
결과 보고는 3-fold 평균 ± std로.

### 4.7 Pseudo-novel에서의 평가 방식

```
각 fold에서:
  1. Train categories로 모델 학습 (base training)
  2. Pseudo-novel 33개 중 1개를 선택, 해당 category에서 1장을 support로 설정
  3. 나머지 해당 category 이미지를 query로 사용
  4. 1-shot으로 detection + segmentation 수행
  5. pycocotools로 mAP, AP50 계산

Private-like pseudo-novel에서의 성능 vs generic pseudo-novel에서의 성능을
별도로 보고하면, implicit privacy gap이 실제로 존재하는지 정량적으로 확인 가능.
```

---

## 5. Baseline 재현 계획

### 5.1 LLM2Seg 재현 (비교군)

```
GPT-4.1 → Qwen3-VL-8B로 대체 (유료 API 사용 불가)
나머지 동일: Grounding DINO + SAM

Steps:
  1. Support 16장 augmentation (shift, crop, rotate, brightness)
  2. Augmented support로 Grounding DINO fine-tune
  3. Query image → Real-ESRGAN → Qwen3-VL category inference → G-DINO → SAM
  4. 3-fold pseudo-novel에서 평가

이 결과가 Track A, Track B의 비교 기준선이 됨.
```

### 5.2 Augmentation 전략

LLM2Seg 논문에 디테일 미공개. Standard augmentation으로 구현:

```
- Random shift: ±10-20% of image size
- Random crop: 80-100% of original
- Random rotation: ±15°
- Brightness adjustment: ±20%
- Category당 50-100장 목표 (augmentation ratio는 실험으로 결정)
```

---

## 6. 실험 로드맵

전체 프로세스는 세 단계로 명확히 구분된다. 각 단계의 목적이 다르므로 섞이면 안 된다.

```
Step 1: 진단      → "이 benchmark가 뭘 측정하는가" 파악 (1회성)
Step 2: 방법론 설계 → 진단 결과에 기반한 Track A/B method 확정
Step 3: 챌린지 수행 → 확정된 method로 학습 + 검증 + 제출
```

---

### Step 1: Benchmark 진단 (반나절, 1회성)

**목적:** 이 benchmark의 information flow를 분해하여, semantic signal / visual signal / detector prior 중 어떤 것이 localization을 지배하는지 파악한다.

**진단 결과에 따라 Step 2의 방향이 결정되므로, 이 단계를 건너뛰면 안 된다.**

```
데이터:    Support 16장 (GT 있음, 유일하게 private category에 대해 정량 평가 가능)
모델:      Pretrained G-DINO + Qwen3-VL-8B (학습 없음)
소요 시간: 반나절

실험 4개:
  Exp4: G-DINO + "object"              → detector prior baseline (floor)
  Exp1: G-DINO + oracle category name  → semantic upper bound
  Exp2: G-DINO + VLM visual description (category 언급 금지) → visual semantic signal
  Exp3: G-DINO + VLM category inference → LLM2Seg style reproduction

평가: category별 best IoU + Hit Rate @IoU≥0.5
상세 가이드: diagnostic_experiment_guide_v2.md 참조
```

**결과 해석:**

```
Case A: Exp1 ≈ Exp3 >> Exp2 >> Exp4
  → Category token이 결정적. Track A에서 VLM category inference 품질 강화 집중.

Case B: Exp1 ≈ Exp2 ≈ Exp3 >> Exp4
  → Semantic signal이면 뭐든 작동. Track B (VLM-native few-shot) 유망.

Case C: Exp1 >> Exp3 >> Exp2 >> Exp4
  → Oracle vs VLM inference 간 gap이 큼. Category inference 정확도가 핵심 변수.

Case D: Exp4 ≈ Exp1 ≈ Exp2 ≈ Exp3
  → Detector prior가 이미 강함. Fine-tune만으로 충분할 수 있음.
```

**이 단계의 산출물:** 4개 실험의 16 category × IoU 테이블 + Case 판정 → Step 2 방향 결정 근거.

---

### Step 2: 방법론 설계 (Step 1 결과 이후)

**목적:** 진단 결과에 기반하여 Track A / Track B의 구체적 method를 확정한다.

```
입력: Step 1의 Case 판정
출력: Track A / Track B 각각의 구체적 학습 전략, 데이터 구성, inference pipeline

여기서 결정해야 할 것들:
  - Track A: G-DINO fine-tune 전략 (base only? base + augmented support?)
  - Track A: VLM의 역할 범위 (category inference? description? reranking?)
  - Track B: 학습 데이터 format (RefCOCO? support-query pair? episodic?)
  - Track B: Support conditioning 방식 (in-context? adapter?)
  - 공통: Augmentation 전략, threshold 설정 등
```

**이 단계에서는 아직 대규모 학습을 돌리지 않는다.** 설계만 확정하고 Step 3으로 넘어간다.

---

### Step 3: 챌린지 수행

Step 2에서 확정된 method를 실행한다. 네 개 phase로 구성.

#### Phase 3-1: 기반 구축 (Week 1-2)

```
- [ ] Augmentation 코드 구현
- [ ] LLM2Seg baseline 재현 (Qwen3-VL로 GPT 대체)
- [ ] 3-fold split 코드 구현 및 데이터 준비
- [ ] Baseline 결과 확인 → 비교 기준선 확보
```

#### Phase 3-2: Track A 실험 (Week 2-4)

```
- [ ] G-DINO를 VizWiz-FewShot base로 fine-tune
- [ ] Step 2에서 확정된 VLM 활용 전략 구현
- [ ] 3-fold pseudo-novel에서 평가 (method 검증 도구)
- [ ] Private-like vs generic pseudo-novel 성능 비교
```

#### Phase 3-3: Track B 실험 (Week 3-5)

```
- [ ] VizWiz-FewShot base를 확정된 format으로 변환
- [ ] Qwen3-VL-8B LoRA fine-tune
- [ ] 3-fold pseudo-novel에서 평가 (method 검증 도구)
```

#### Phase 3-4: 분석 및 제출 (Week 5-6)

```
- [ ] Track A vs Track B vs Baseline 비교 분석
- [ ] 최종 모델로 실제 query set 1,056장 inference
- [ ] Submission JSON 생성 및 제출
- [ ] Extended abstract 작성 (2 pages, CVPR format)
```

---

### 도구의 목적 구분

```
Support 16장:
  → Step 1 (진단)에서 사용
  → Benchmark의 본질을 파악하는 1회성 분석 도구
  → 이후 Step 3에서는 augmentation + inference reference로만 사용

3-fold pseudo-novel:
  → Step 3 (챌린지 수행)에서 사용
  → Method를 검증하는 반복적 evaluation 도구
  → 진단 도구가 아님. Method가 확정된 후에 사용.
```

---

## 7. Extended Abstract Framing (Draft)

```
핵심 메시지:

"We interpret the VizWiz FSOL challenge not as plain few-shot detection,
but as region-level implicit privacy-aware discrimination, where a model
trained on non-private categories must localize private novel objects
from 1-shot support references.

Our approach differs from the prior top-performing method (LLM2Seg) in two key ways:
(1) we actively utilize the officially provided VizWiz-FewShot base classes
for training, which LLM2Seg did not leverage, and
(2) we design support-conditioned inference where the support image's
visual information directly participates in the detection process,
rather than relying solely on VLM prior knowledge.

We present results from two complementary tracks:
Track A (grounding-centric hybrid) maximizes detection performance,
while Track B (VLM-centric single model) explores whether a single VLM
can perform implicit privacy-aware grounding through in-context few-shot learning."
```

---

## 8. 연구 연결 매핑

```
이 챌린지에서의 작업 → 세 군데에 동시 활용:

[신진연구 과제]
  - 세부목표(나): LVLM 위치 기반 deep reasoning, mAP 65% 이상
  - 세부목표(다): open-vocabulary reasoning
  - BIV-Priv-Seg 세계최고(61.63%) 초과 달성 = 1차년도 핵심 성과

[Paper 1 (AAAI 2026, ~8월)]
  - Track B 결과 = 단일 VLM의 grounding-reasoning consistency 실험 데이터
  - Track A vs B 비교 = "모듈 파이프라인에서는 alignment 검증 불가" 실증
  - Thinking mode reasoning chain = consistency 분석 데이터

[VizWiz 2026 Workshop]
  - 챌린지 제출 (Track A or B, 높은 쪽)
  - Extended abstract 제출
  - Danna Gurari 그룹과의 접점 형성
```
