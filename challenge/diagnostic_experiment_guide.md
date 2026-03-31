# Diagnostic Experiment Guide v2 — Benchmark 분석 실험

---

## 실험 목적

```
이 실험은 mAP를 올리기 위한 실험이 아니다.
이 benchmark에서 localization을 지배하는 signal이 무엇인지를 먼저 확인하는 실험이다.

핵심 질문: support signal이 실제로 query localization에 필요한가?

이걸 먼저 알아야 하는 이유:
  Track A (G-DINO hybrid)와 Track B (Qwen3-VL single) 중
  어디에 시간을 써야 하는지 아직 모른다.

가능한 해석:
  1) category name만 맞으면 찾는다
     → 핵심은 VLM category inference
  2) visual description만으로도 찾는다
     → support-conditioned approach가 유망
  3) category name을 줘도 못 찾는다
     → detector 한계가 핵심, G-DINO fine-tune 우선
  4) "object"만 줘도 잘 찾는다
     → semantic signal보다 detector prior/fine-tune이 더 중요

이 네 가지를 추측이 아니라 데이터로 분리하는 것이 목적이다.

실험 구조:
  support image → semantic signal 생성 (여기서 prompt가 만들어짐)
  query image → localization (여기서 detection이 일어남)
  이 둘을 분리해야 한다.
```

---

## 실험 구조 (4개)

```
Exp1: G-DINO + oracle category name       → upper bound
Exp2: G-DINO + visual description          → semantic grounding (category 언급 금지)
Exp3: G-DINO + VLM category inference      → LLM2Seg reproduction
Exp4: G-DINO + "object"                    → detector prior baseline
```

**모든 실험에서:**
- Prompt는 **support image**에서 생성
- Detection은 **query image**에서 수행
- IoU는 query image의 GT와 비교

이 4개 실험은 method 자체가 아니라 의사결정용 진단이다.
결과 해석은 다음 질문으로 연결되어야 한다:
  - semantic bottleneck인가?
  - detector bottleneck인가?
  - support visual information이 실제로 필요한가?
  - classification 강화와 detector fine-tuning 중 어디에 힘을 줘야 하는가?

---

## 0. 사전 준비

```python
import json
from collections import defaultdict
from groundingdino.util.inference import load_model, load_image, predict

# G-DINO 로드 (pretrained, fine-tune 안 한 원본)
model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth"
)

# Support set 로드
with open('support_set.json', 'r') as f:
    support_data = json.load(f)

# Category 매핑
PRIVATE_CATEGORIES = {c['id']: c['name'] for c in support_data['categories']}

# Support image별 GT 정리
support_gt = {}
for ann in support_data['annotations']:
    support_gt[ann['image_id']] = {
        'category_id': ann['category_id'],
        'category_name': PRIVATE_CATEGORIES[ann['category_id']],
        'bbox': ann['bbox']  # [x, y, w, h] COCO format
    }

# ============================================================
# 중요: Query image 준비
# 가장 좋은 선택:
#   - GT가 있는 dev-158 또는 pseudo-novel query_eval 사용
# 차선:
#   - support 16장 leave-one-out self-detection
#
# 단, self-detection은 빠른 smoke/format check용이다.
# 이 결과만으로 "어떤 signal이 중요하다"는 결론을 내리면 안 된다.
# support와 query가 동일 이미지면 detector prior가 과대평가될 수 있다.
# 의사결정용 비교는 반드시 별도 query 이미지에서 다시 확인해야 한다.
# ============================================================

# 구현 메모:
#   아래 코드 블록은 최소 예시다.
#   실제 실행에서는
#     - support image: prompt 생성용
#     - query image: detection/IoU 평가용
#   으로 분리해야 한다.
#   즉 loop는 query set 기준으로 돌고,
#   support는 해당 category의 reference로만 사용해야 한다.
```

---

## 1. Bbox Format 변환 (가장 먼저 확인)

```python
def gdino_to_coco_bbox(gdino_box, img_w, img_h):
    """
    G-DINO output: [cx, cy, w, h] normalized (0~1)
    COCO GT:       [x, y, w, h] pixel
    """
    cx, cy, w, h = gdino_box
    x = (cx - w/2) * img_w
    y = (cy - h/2) * img_h
    w = w * img_w
    h = h * img_h
    return [x, y, w, h]

def compute_iou(box1, box2):
    """
    box1, box2: [x, y, w, h] (COCO format, pixel)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0]+box1[2], box2[0]+box2[2])
    y2 = min(box1[1]+box1[3], box2[1]+box2[3])
    
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0
```

---

## 2. Exp4: Detector Prior Baseline

**가장 먼저 돌려라. 이게 floor.**

```python
BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.15

results_exp4 = {}

for img_id, gt in support_gt.items():
    img_path = f"support_images/{get_filename(img_id)}"  # 파일명 매핑
    image_source, image = load_image(img_path)
    img_h, img_w = image_source.shape[:2]
    
    # "object"라는 generic prompt만 사용
    boxes, logits, phrases = predict(
        model=model, image=image,
        caption="object",
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    
    # bbox 변환 + best IoU 계산
    best_iou = 0
    for box in boxes:
        pred_bbox = gdino_to_coco_bbox(box.tolist(), img_w, img_h)
        iou = compute_iou(gt['bbox'], pred_bbox)
        best_iou = max(best_iou, iou)
    
    results_exp4[img_id] = {
        'category': gt['category_name'],
        'n_detected': len(boxes),
        'top_score': logits[0].item() if len(logits) > 0 else 0,
        'best_iou': best_iou,
        'hit': best_iou >= 0.5
    }
```

---

## 3. Exp1: Oracle Category Name (Upper Bound)

```python
results_exp1 = {}

for img_id, gt in support_gt.items():
    img_path = f"support_images/{get_filename(img_id)}"
    image_source, image = load_image(img_path)
    img_h, img_w = image_source.shape[:2]
    
    # Oracle: GT category name을 직접 사용
    boxes, logits, phrases = predict(
        model=model, image=image,
        caption=gt['category_name'],    # 예: "bank statement"
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    
    best_iou = 0
    for box in boxes:
        pred_bbox = gdino_to_coco_bbox(box.tolist(), img_w, img_h)
        iou = compute_iou(gt['bbox'], pred_bbox)
        best_iou = max(best_iou, iou)
    
    results_exp1[img_id] = {
        'category': gt['category_name'],
        'n_detected': len(boxes),
        'top_score': logits[0].item() if len(logits) > 0 else 0,
        'best_iou': best_iou,
        'hit': best_iou >= 0.5
    }
```

---

## 4. Exp2: VLM Visual Description (Category 언급 금지)

```python
results_exp2 = {}

DESCRIPTION_PROMPT = """Look at the main object in this image.
Describe it ONLY by its physical appearance.
Include: shape, color, size, texture, any visible text or markings, material.
Do NOT name what the object is. Do NOT mention its category or function.
Keep your description to 1-2 sentences maximum."""

for img_id, gt in support_gt.items():
    img_path = f"support_images/{get_filename(img_id)}"
    image_source, image = load_image(img_path)
    img_h, img_w = image_source.shape[:2]
    
    # Step 1: Support image → VLM description (category 언급 금지)
    description = qwen_inference(img_path, DESCRIPTION_PROMPT)
    # ↑ 네 Qwen3-VL inference 함수로 대체
    
    # Step 2: Description → G-DINO
    boxes, logits, phrases = predict(
        model=model, image=image,
        caption=description,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    
    best_iou = 0
    for box in boxes:
        pred_bbox = gdino_to_coco_bbox(box.tolist(), img_w, img_h)
        iou = compute_iou(gt['bbox'], pred_bbox)
        best_iou = max(best_iou, iou)
    
    results_exp2[img_id] = {
        'category': gt['category_name'],
        'description': description,
        'n_detected': len(boxes),
        'top_score': logits[0].item() if len(logits) > 0 else 0,
        'best_iou': best_iou,
        'hit': best_iou >= 0.5
    }
    
    print(f"[Exp2] {gt['category_name']}")
    print(f"  Desc: {description[:120]}...")
    print(f"  IoU: {best_iou:.3f}")
```

---

## 5. Exp3: VLM Category Inference (LLM2Seg Style)

```python
results_exp3 = {}

CATEGORY_LIST = "\n".join([f"- {name}" for name in PRIVATE_CATEGORIES.values()])

CLASSIFY_PROMPT = f"""Which category does the main object in this image belong to?

Choose from:
{CATEGORY_LIST}

Return only the category name. Nothing else."""

for img_id, gt in support_gt.items():
    img_path = f"support_images/{get_filename(img_id)}"
    image_source, image = load_image(img_path)
    img_h, img_w = image_source.shape[:2]
    
    # Step 1: VLM category inference
    vlm_output = qwen_inference(img_path, CLASSIFY_PROMPT)
    inferred_category = vlm_output.strip()
    category_correct = gt['category_name'].lower() in inferred_category.lower()
    
    # Step 2: Inferred category → G-DINO
    boxes, logits, phrases = predict(
        model=model, image=image,
        caption=inferred_category,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )
    
    best_iou = 0
    for box in boxes:
        pred_bbox = gdino_to_coco_bbox(box.tolist(), img_w, img_h)
        iou = compute_iou(gt['bbox'], pred_bbox)
        best_iou = max(best_iou, iou)
    
    results_exp3[img_id] = {
        'category': gt['category_name'],
        'inferred': inferred_category,
        'category_correct': category_correct,
        'n_detected': len(boxes),
        'top_score': logits[0].item() if len(logits) > 0 else 0,
        'best_iou': best_iou,
        'hit': best_iou >= 0.5
    }
    
    print(f"[Exp3] GT: {gt['category_name']:<30} Pred: {inferred_category:<30} {'✓' if category_correct else '✗'}")
```

---

## 6. 결과 출력

```python
def print_results_table(exp1, exp2, exp3, exp4):
    print(f"\n{'='*110}")
    print(f"  Diagnostic Results: Benchmark Analysis")
    print(f"{'='*110}")
    print(f"{'Category':<28} {'Exp4(obj)':>10} {'Exp1(oracle)':>12} {'Exp2(desc)':>10} {'Exp3(infer)':>11} {'Cat OK':>7}")
    print(f"{' '*28} {'IoU':>10} {'IoU':>12} {'IoU':>10} {'IoU':>11} {'':>7}")
    print(f"{'-'*110}")
    
    for img_id in exp1.keys():
        cat = exp1[img_id]['category']
        iou4 = exp4[img_id]['best_iou']
        iou1 = exp1[img_id]['best_iou']
        iou2 = exp2[img_id]['best_iou']
        iou3 = exp3[img_id]['best_iou']
        cat_ok = '✓' if exp3[img_id]['category_correct'] else '✗'
        
        print(f"{cat:<28} {iou4:>10.3f} {iou1:>12.3f} {iou2:>10.3f} {iou3:>11.3f} {cat_ok:>7}")
    
    # Summary
    print(f"{'-'*110}")
    for name, results in [("Exp4(object)", exp4), ("Exp1(oracle)", exp1), 
                           ("Exp2(desc)", exp2), ("Exp3(infer)", exp3)]:
        hits = sum(1 for r in results.values() if r['hit'])
        avg_iou = sum(r['best_iou'] for r in results.values()) / len(results)
        print(f"  {name:<20} Hit@50: {hits}/16 ({hits/16*100:.1f}%)  Mean IoU: {avg_iou:.3f}")
    
    cat_acc = sum(1 for r in exp3.values() if r['category_correct'])
    print(f"  {'Exp3 Cat Acc':<20} {cat_acc}/16 ({cat_acc/16*100:.1f}%)")

print_results_table(results_exp1, results_exp2, results_exp3, results_exp4)
```

---

## 6.5. 최소 보고 항목

이 실험은 16개 category별 raw 결과보다 아래 집계가 더 중요하다.

```python
# category-wise table
# summary: mean IoU / Hit@50
# gap metrics:
#   delta_oracle_vs_infer = Exp1 - Exp3
#   delta_oracle_vs_desc  = Exp1 - Exp2
#   delta_oracle_vs_obj   = Exp1 - Exp4
#
# 해석:
#   Exp1 - Exp3 크면 category inference loss가 큼
#   Exp1 - Exp2 작으면 visual description도 semantic signal로 충분
#   Exp1 - Exp4 작으면 semantic signal보다 detector prior가 큼
```

최소 산출물:
- category별 `best_iou`, `hit@50`, `n_detected`
- experiment별 평균 IoU와 Hit@50
- `Exp1-Exp3`, `Exp1-Exp2`, `Exp1-Exp4` gap
- 실패 케이스 3~5개 시각화

---

## 7. 결과 해석 가이드

```
Case A: Exp1 ≈ Exp3 >> Exp2 >> Exp4
  → Category token이 결정적. Description은 부족.
  → LLM의 category inference 품질이 핵심 변수.
  → 우리 전략: Track A에서 VLM category inference 강화에 집중.

Case B: Exp1 ≈ Exp2 ≈ Exp3 >> Exp4
  → Semantic signal이면 뭐든 작동. Open vocabulary detector.
  → Description만으로도 충분 = support visual info가 유용함.
  → 우리 전략: Track B (VLM-native few-shot)가 유망.

Case C: Exp1 >> Exp3 >> Exp2 >> Exp4
  → Oracle category가 압도적. VLM inference에서 손실 큼.
  → Semantic label quality가 핵심.
  → 우리 전략: Category inference 정확도 올리기에 집중.

Case D: Exp4 ≈ Exp1 ≈ Exp2 ≈ Exp3
  → Detector prior가 이미 강함. Semantic signal 무관.
  → 이 benchmark는 사실 detector 성능 문제.
  → 가능성 낮지만, 이러면 fine-tune만으로 충분.

Case E: Exp4 >> 나머지
  → Generic "object" detection이 category-specific보다 나음.
  → Text prompt가 오히려 방해. 드문 케이스.
```

---

## 8. 주의사항 체크리스트

- [ ] G-DINO bbox format 변환 확인 (normalized cxcywh → pixel xywh)
- [ ] 모든 실험에서 동일 threshold 사용 (BOX: 0.15, TEXT: 0.15)
- [ ] Exp2 description에 category name이 누출되지 않는지 확인
- [ ] Exp3 classification prompt가 단순하고 noise 없는지 확인
- [ ] Support image 파일 경로 매핑 확인
- [ ] 결과를 category별 + summary 두 레벨로 기록
- [ ] 가능하면 self-detection이 아니라 별도 query image로 비교
- [ ] 4개 실험 모두 동일한 query image 집합에서 수행
- [ ] prompt만 바뀌고 detector/threshold/query는 동일한지 확인
- [ ] 이 실험 결과를 challenge 최종 성능처럼 해석하지 않기

---

## 9. 바로 실행할 스크립트

구현된 실행 스크립트:
- [`challenge/scripts/run_benchmark_signal_diagnostic.py`](/home/choheeseung/workspace/vlm-privacy/challenge/scripts/run_benchmark_signal_diagnostic.py)

예시:

```bash
conda activate psi
python challenge/scripts/run_benchmark_signal_diagnostic.py \
  --support-json /home/choheeseung/workspace/vlm-privacy/challenge/results/pseudo_novel_splits/split1/support_1shot.json \
  --support-image-dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/images \
  --query-json /home/choheeseung/workspace/vlm-privacy/challenge/results/pseudo_novel_splits/split1/query_eval.json \
  --query-image-dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/images \
  --output-dir /home/choheeseung/workspace/vlm-privacy/challenge/results/diagnostic_signal/split1 \
  --config-path /home/choheeseung/workspace/vlm-privacy/challenge/LLM2Seg/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py \
  --checkpoint-path /home/choheeseung/workspace/vlm-privacy/challenge/LLM2Seg/checkpoints/groundingdino_swint_ogc_mmdet-822d7e9d.pth \
  --llm-model /home/choheeseung/workspace/vlm-privacy/challenge/models/Qwen3-VL-4B-Instruct \
  --device cuda
```

산출물:
- `prompts/category_prompts.json`
- `diagnostic_records.json`
- `diagnostic_summary.json`
