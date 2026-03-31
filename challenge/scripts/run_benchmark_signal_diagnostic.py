#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Any

import mmcv


REPO_ROOT = Path(__file__).resolve().parents[2]
LLM2SEG_DIR = Path(os.environ.get('LLM2SEG_DIR', REPO_ROOT / 'third_party' / 'LLM2Seg'))
if str(LLM2SEG_DIR) not in sys.path:
    sys.path.insert(0, str(LLM2SEG_DIR))

from call_vlm import SwiftVLMCaller  # noqa: E402
from models import load_groundingdino_model  # noqa: E402
from utils import preprocess_caption  # noqa: E402


DESCRIPTION_PROMPT = """Look at the main object in this image.
Describe it ONLY by its physical appearance.
Include: shape, color, size, texture, any visible text or markings, material.
Do NOT name what the object is. Do NOT mention its category or function.
Keep your description to 1-2 sentences maximum."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the four-way benchmark-signal diagnostic: oracle / description / inferred / object."
    )
    parser.add_argument("--support-json", required=True, help="COCO JSON containing 1-shot support images.")
    parser.add_argument("--support-image-dir", required=True, help="Directory containing support images.")
    parser.add_argument("--query-json", required=True, help="COCO JSON containing query images and GT boxes.")
    parser.add_argument("--query-image-dir", required=True, help="Directory containing query images.")
    parser.add_argument("--output-dir", required=True, help="Root directory for outputs.")
    parser.add_argument("--config-path", required=True, help="Grounding DINO config path.")
    parser.add_argument("--checkpoint-path", required=True, help="Grounding DINO checkpoint path.")
    parser.add_argument("--llm-model", required=True, help="Local Qwen model directory for swift.")
    parser.add_argument("--device", default="cuda", help="Device for G-DINO and Qwen.")
    parser.add_argument("--box-threshold", type=float, default=0.15, help="Score threshold for kept boxes.")
    parser.add_argument("--text-threshold", type=float, default=0.15, help="Reserved for compatibility/reporting.")
    parser.add_argument("--llm-max-new-tokens", type=int, default=192, help="Qwen max new tokens.")
    parser.add_argument("--llm-decoding-mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--llm-seed", type=int, default=None)
    parser.add_argument("--llm-max-pixels", type=int, default=448)
    parser.add_argument("--date-tag", default=None, help="Optional YYYYMMDD override.")
    parser.add_argument("--run-id", default=None, help="Optional run id override.")
    parser.add_argument("--flat-output", action="store_true", help="Write directly into output-dir without date/run.")
    parser.add_argument("--category-limit", type=int, default=None, help="Optional category cap for quick checks.")
    parser.add_argument("--query-limit", type=int, default=None, help="Optional image-category pair cap for quick checks.")
    return parser.parse_args()


@dataclass
class QueryTask:
    image_id: int
    file_name: str
    category_id: int
    category_name: str
    gt_bboxes: list[list[float]]


def coco_xywh_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0.0, box1[2]) * max(0.0, box1[3])
    area2 = max(0.0, box2[2]) * max(0.0, box2[3])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.flat_output:
        out_dir = Path(args.output_dir)
    else:
        now = datetime.now()
        date_tag = args.date_tag or now.strftime("%Y%m%d")
        run_id = args.run_id or now.strftime("%H%M%S")
        out_dir = Path(args.output_dir) / date_tag / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_coco(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_category_name(name: str) -> str:
    return name.replace("_", " ").strip()


def choose_support_per_category(support_coco: dict[str, Any]) -> dict[int, dict[str, Any]]:
    images_by_id = {img["id"]: img for img in support_coco["images"]}
    anns_by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in support_coco["annotations"]:
        anns_by_cat[ann["category_id"]].append(ann)

    support_by_cat: dict[int, dict[str, Any]] = {}
    for category_id, anns in anns_by_cat.items():
        chosen = sorted(
            anns,
            key=lambda ann: (-float(ann.get("area", 0.0)), ann["image_id"], ann["id"]),
        )[0]
        support_by_cat[category_id] = {
            "annotation": chosen,
            "image": images_by_id[chosen["image_id"]],
        }
    return support_by_cat


def build_query_tasks(query_coco: dict[str, Any], allowed_category_ids: set[int]) -> list[QueryTask]:
    images_by_id = {img["id"]: img for img in query_coco["images"]}
    categories_by_id = {cat["id"]: normalize_category_name(cat["name"]) for cat in query_coco["categories"]}
    anns_by_pair: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for ann in query_coco["annotations"]:
        if ann["category_id"] not in allowed_category_ids:
            continue
        anns_by_pair[(ann["image_id"], ann["category_id"])].append(ann)

    tasks: list[QueryTask] = []
    for (image_id, category_id), anns in anns_by_pair.items():
        image = images_by_id[image_id]
        tasks.append(
            QueryTask(
                image_id=image_id,
                file_name=image["file_name"],
                category_id=category_id,
                category_name=categories_by_id[category_id],
                gt_bboxes=[ann["bbox"] for ann in anns],
            )
        )
    tasks.sort(key=lambda task: (task.category_id, task.image_id))
    return tasks


def build_category_prompt(category_names: list[str]) -> str:
    category_list = "\n".join(f"- {name}" for name in category_names)
    return (
        "Which category does the main object in this image belong to?\n\n"
        "Choose from:\n"
        f"{category_list}\n\n"
        "Return only the category name. Nothing else."
    )


def strip_output_tags(text: str) -> str:
    match = re.search(r"<output>(.*?)</output>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_inferred_category(raw_text: str, allowed_names: list[str]) -> tuple[str, bool]:
    cleaned = strip_output_tags(raw_text).strip().strip("[]")
    lowered = cleaned.lower()
    lowered_names = {name.lower(): name for name in allowed_names}
    if lowered in lowered_names:
        return lowered_names[lowered], True

    for name in allowed_names:
        if name.lower() in lowered:
            return name, True

    compact = re.sub(r"[^a-z0-9]+", " ", lowered).strip()
    for name in allowed_names:
        if re.sub(r"[^a-z0-9]+", " ", name.lower()).strip() == compact:
            return name, True

    return cleaned, False


def has_category_leak(text: str, category_name: str) -> bool:
    normalized_text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    normalized_category = re.sub(r"[^a-z0-9]+", " ", category_name.lower()).strip()
    return bool(normalized_category) and normalized_category in normalized_text


def load_qwen(args: argparse.Namespace) -> SwiftVLMCaller:
    return SwiftVLMCaller(
        model_path=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
    )


def extract_pred_boxes(result: Any, score_threshold: float) -> tuple[list[list[float]], list[float]]:
    if isinstance(result, list):
        if not result:
            return [], []
        result = result[0]

    predictions = result.get("predictions", []) if isinstance(result, dict) else []
    if not predictions:
        return [], []
    prediction = predictions[0]
    boxes = prediction.get("bboxes", [])
    scores = prediction.get("scores", [])
    kept_boxes: list[list[float]] = []
    kept_scores: list[float] = []
    for box, score in zip(boxes, scores):
        score_value = float(score)
        if score_value < score_threshold:
            continue
        kept_boxes.append([float(x) for x in box])
        kept_scores.append(score_value)
    return kept_boxes, kept_scores


def best_iou_against_targets(pred_boxes_xyxy: list[list[float]], gt_boxes_xywh: list[list[float]]) -> tuple[float, list[float] | None]:
    best_iou = 0.0
    best_box = None
    for box in pred_boxes_xyxy:
        pred = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
        for gt in gt_boxes_xywh:
            iou = coco_xywh_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_box = pred
    return best_iou, best_box


def run_detector(model: Any, image_path: Path, prompt: str, score_threshold: float) -> dict[str, Any]:
    image = mmcv.imread(str(image_path), channel_order="rgb")
    result = model(inputs=image, texts=[preprocess_caption(prompt)])
    boxes, scores = extract_pred_boxes(result, score_threshold=score_threshold)
    return {
        "prompt": prompt,
        "num_boxes": len(boxes),
        "top_score": max(scores) if scores else 0.0,
        "boxes_xyxy": boxes,
    }


def summarize(records: list[dict[str, Any]], experiment: str) -> dict[str, Any]:
    subset = [record for record in records if record["experiment"] == experiment]
    if not subset:
        return {
            "count": 0,
            "mean_iou": 0.0,
            "hit_at_50": 0.0,
            "mean_top_score": 0.0,
        }
    count = len(subset)
    return {
        "count": count,
        "mean_iou": sum(record["best_iou"] for record in subset) / count,
        "hit_at_50": sum(1 for record in subset if record["best_iou"] >= 0.5) / count,
        "mean_top_score": sum(record["top_score"] for record in subset) / count,
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args)
    (output_dir / "prompts").mkdir(parents=True, exist_ok=True)
    print(f"[setup] output_dir={output_dir}", flush=True)

    support_coco = load_coco(args.support_json)
    query_coco = load_coco(args.query_json)

    support_by_cat = choose_support_per_category(support_coco)
    category_id_to_name = {cat["id"]: normalize_category_name(cat["name"]) for cat in support_coco["categories"]}
    category_ids = sorted(support_by_cat.keys())
    if args.category_limit is not None:
        category_ids = category_ids[: args.category_limit]
    category_names = [category_id_to_name[cat_id] for cat_id in category_ids]
    print(
        f"[setup] support_categories={len(category_ids)} "
        f"query_images={len(query_coco['images'])} query_annotations={len(query_coco['annotations'])}",
        flush=True,
    )

    query_tasks = build_query_tasks(query_coco, set(category_ids))
    if args.query_limit is not None:
        query_tasks = query_tasks[: args.query_limit]
    print(f"[setup] query_tasks={len(query_tasks)}", flush=True)

    print("[setup] loading Qwen swift backend...", flush=True)
    qwen = load_qwen(args)
    print("[setup] loading Grounding DINO...", flush=True)
    gdino = load_groundingdino_model(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )
    print("[setup] models ready", flush=True)

    classification_prompt = build_category_prompt(category_names)
    category_prompts: dict[int, dict[str, Any]] = {}
    prompt_manifest = output_dir / "prompts" / "category_prompts.json"
    for index, category_id in enumerate(category_ids, start=1):
        support_info = support_by_cat[category_id]
        category_name = category_id_to_name[category_id]
        support_image_path = Path(args.support_image_dir) / support_info["image"]["file_name"]
        print(
            f"[support {index}/{len(category_ids)}] category={category_name} "
            f"support_image={support_info['image']['file_name']} -> description",
            flush=True,
        )
        description_raw = qwen.generate(str(support_image_path), DESCRIPTION_PROMPT)
        print(
            f"[support {index}/{len(category_ids)}] category={category_name} -> category_inference",
            flush=True,
        )
        inferred_raw = qwen.generate(str(support_image_path), classification_prompt)
        inferred_category, inferred_valid = parse_inferred_category(inferred_raw, category_names)
        prompt_record = {
            "category_id": category_id,
            "category_name": category_name,
            "support_image_id": support_info["image"]["id"],
            "support_file_name": support_info["image"]["file_name"],
            "oracle_prompt": category_name,
            "description_prompt": strip_output_tags(description_raw),
            "description_has_category_leak": has_category_leak(strip_output_tags(description_raw), category_name),
            "inferred_raw": inferred_raw,
            "inferred_prompt": inferred_category,
            "inferred_valid": inferred_valid,
            "category_correct": inferred_category.lower() == category_name.lower(),
            "object_prompt": "object",
        }
        category_prompts[category_id] = prompt_record
        prompt_manifest.write_text(json.dumps(category_prompts, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[support {index}/{len(category_ids)}] saved prompt meta "
            f"(inferred={inferred_category!r}, valid={inferred_valid}, "
            f"desc_leak={prompt_record['description_has_category_leak']})",
            flush=True,
        )

    prompt_manifest.write_text(json.dumps(category_prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[support] prompt manifest saved to {prompt_manifest}", flush=True)

    experiments = {
        "exp1_oracle": lambda meta: meta["oracle_prompt"],
        "exp2_description": lambda meta: meta["description_prompt"],
        "exp3_inferred": lambda meta: meta["inferred_prompt"],
        "exp4_object": lambda meta: meta["object_prompt"],
    }

    records: list[dict[str, Any]] = []
    records_jsonl_path = output_dir / "diagnostic_records.jsonl"
    if records_jsonl_path.exists():
        records_jsonl_path.unlink()
    total_runs = len(query_tasks) * len(experiments)
    run_counter = 0
    for task_index, task in enumerate(query_tasks, start=1):
        prompt_meta = category_prompts[task.category_id]
        image_path = Path(args.query_image_dir) / task.file_name
        print(
            f"[query {task_index}/{len(query_tasks)}] image={task.file_name} "
            f"category={task.category_name} gt_boxes={len(task.gt_bboxes)}",
            flush=True,
        )
        for experiment_name, prompt_getter in experiments.items():
            prompt = prompt_getter(prompt_meta)
            detection = run_detector(gdino, image_path, prompt, score_threshold=args.box_threshold)
            best_iou, best_box = best_iou_against_targets(detection["boxes_xyxy"], task.gt_bboxes)
            record = {
                "experiment": experiment_name,
                "image_id": task.image_id,
                "file_name": task.file_name,
                "category_id": task.category_id,
                "category_name": task.category_name,
                "gt_bboxes": task.gt_bboxes,
                "prompt": prompt,
                "num_boxes": detection["num_boxes"],
                "top_score": detection["top_score"],
                "best_iou": best_iou,
                "hit_at_50": best_iou >= 0.5,
                "best_pred_bbox_xywh": best_box,
            }
            if experiment_name == "exp3_inferred":
                record["inferred_raw"] = prompt_meta["inferred_raw"]
                record["inferred_valid"] = prompt_meta["inferred_valid"]
                record["category_correct"] = prompt_meta["category_correct"]
            records.append(record)
            with records_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            run_counter += 1
            print(
                f"  [{experiment_name}] boxes={detection['num_boxes']} "
                f"top_score={detection['top_score']:.4f} best_iou={best_iou:.4f} "
                f"progress={run_counter}/{total_runs}",
                flush=True,
            )

    summaries = {name: summarize(records, name) for name in experiments}
    exp1_iou = summaries["exp1_oracle"]["mean_iou"]
    summary = {
        "support_json": args.support_json,
        "query_json": args.query_json,
        "num_categories": len(category_ids),
        "num_query_tasks": len(query_tasks),
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "experiments": summaries,
        "gaps": {
            "exp1_minus_exp3_mean_iou": exp1_iou - summaries["exp3_inferred"]["mean_iou"],
            "exp1_minus_exp2_mean_iou": exp1_iou - summaries["exp2_description"]["mean_iou"],
            "exp1_minus_exp4_mean_iou": exp1_iou - summaries["exp4_object"]["mean_iou"],
        },
        "exp3_category_accuracy": (
            sum(1 for meta in category_prompts.values() if meta["category_correct"]) / len(category_prompts)
            if category_prompts
            else 0.0
        ),
        "exp2_description_leak_rate": (
            sum(1 for meta in category_prompts.values() if meta["description_has_category_leak"]) / len(category_prompts)
            if category_prompts
            else 0.0
        ),
    }

    (output_dir / "diagnostic_records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved prompts to {prompt_manifest}")
    print(f"Saved incremental per-query records to {records_jsonl_path}")
    print(f"Saved per-query records to {output_dir / 'diagnostic_records.json'}")
    print(f"Saved summary to {output_dir / 'diagnostic_summary.json'}")


if __name__ == "__main__":
    main()
