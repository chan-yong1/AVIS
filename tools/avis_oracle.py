import argparse
import copy
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "avism" / "data", REPO_ROOT / "AVISM" / "data"):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break
else:
    raise FileNotFoundError("Could not locate aviseval package under avism/data or AVISM/data")
import aviseval  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run offline oracle experiments for AVIS.")
    parser.add_argument("--gt", required=True, help="Path to AVIS ground-truth json, e.g. datasets/test.json")
    parser.add_argument("--pred", required=True, help="Path to baseline prediction json (results.json)")
    parser.add_argument("--out-dir", required=True, help="Directory to write oracle trackers and summaries")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["baseline", "mask", "association", "sounding", "full"],
        choices=["baseline", "mask", "association", "sounding", "full"],
        help="Oracle variants to generate and evaluate",
    )
    parser.add_argument(
        "--min-track-iou",
        type=float,
        default=1e-6,
        help="Minimum track IoU to keep a track-wise Hungarian match for mask/sounding oracles",
    )
    parser.add_argument(
        "--min-frame-iou",
        type=float,
        default=1e-6,
        help="Minimum frame IoU to keep a frame-wise match for association oracle",
    )
    return parser.parse_args()


def ensure_serializable_rle(rle):
    rle = copy.deepcopy(rle)
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        rle["counts"] = counts.decode("utf-8")
    return rle


def ensure_rle(segmentation, height, width):
    if not segmentation:
        return None
    if isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            segmentation = mask_utils.frPyObjects(segmentation, height, width)
        return ensure_serializable_rle(segmentation)
    rles = mask_utils.frPyObjects(segmentation, height, width)
    return ensure_serializable_rle(mask_utils.merge(rles))


def encode_empty_rle(height, width):
    rle = mask_utils.encode(np.asfortranarray(np.zeros((height, width, 1), dtype=np.uint8)))[0]
    return ensure_serializable_rle(rle)


def seg_area(segmentation):
    if segmentation is None:
        return 0.0
    return float(np.asarray(mask_utils.area(segmentation)).item())


def is_active(segmentation):
    return seg_area(segmentation) > 0.0


def safe_iou(seg_a, seg_b):
    if seg_a is None or seg_b is None:
        return 0.0
    if not is_active(seg_a) or not is_active(seg_b):
        return 0.0
    return float(mask_utils.iou([seg_a], [seg_b], [False])[0, 0])


def copy_segmentation(segmentation):
    if segmentation is None:
        return None
    return ensure_serializable_rle(segmentation)


def load_gt(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    video_info = {}
    for video in gt_data["videos"]:
        video_info[video["id"]] = {
            "height": video["height"],
            "width": video["width"],
            "length": len(video["file_names"]),
            "name": video["file_names"][0].split("/")[0],
        }

    gt_tracks_by_video = defaultdict(list)
    for ann in gt_data["annotations"]:
        info = video_info[ann["video_id"]]
        segmentations = [ensure_rle(seg, info["height"], info["width"]) for seg in ann["segmentations"]]
        gt_tracks_by_video[ann["video_id"]].append(
            {
                "video_id": ann["video_id"],
                "category_id": ann["category_id"],
                "track_id": ann["id"],
                "segmentations": segmentations,
                "active_frames": {idx for idx, seg in enumerate(segmentations) if is_active(seg)},
            }
        )
    return gt_data, video_info, gt_tracks_by_video


def load_predictions(pred_path, video_info):
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    pred_tracks_by_video = defaultdict(list)
    for local_id, pred in enumerate(pred_data):
        info = video_info[pred["video_id"]]
        segmentations = [ensure_rle(seg, info["height"], info["width"]) for seg in pred["segmentations"]]
        pred_tracks_by_video[pred["video_id"]].append(
            {
                "video_id": pred["video_id"],
                "category_id": pred["category_id"],
                "score": float(pred["score"]),
                "segmentations": segmentations,
                "pred_index": local_id,
                "active_frames": {idx for idx, seg in enumerate(segmentations) if is_active(seg)},
            }
        )
    return pred_data, pred_tracks_by_video


def make_output_track(template, segmentations, score=None, category_id=None):
    return {
        "video_id": template["video_id"],
        "score": float(template["score"] if score is None else score),
        "category_id": int(template["category_id"] if category_id is None else category_id),
        "segmentations": [copy_segmentation(seg) for seg in segmentations],
    }


def make_gt_output_track(gt_track, video_info_item, score=1.0):
    empty_seg = encode_empty_rle(video_info_item["height"], video_info_item["width"])
    full_segmentations = []
    for seg in gt_track["segmentations"]:
        full_segmentations.append(copy_segmentation(seg) if is_active(seg) else copy_segmentation(empty_seg))
    return {
        "video_id": gt_track["video_id"],
        "score": float(score),
        "category_id": int(gt_track["category_id"]),
        "segmentations": full_segmentations,
    }


def build_track_match(gt_tracks, pred_tracks, min_track_iou):
    matches = {}
    gt_by_class = defaultdict(list)
    pred_by_class = defaultdict(list)
    for gt_idx, gt_track in enumerate(gt_tracks):
        gt_by_class[gt_track["category_id"]].append((gt_idx, gt_track))
    for pred_idx, pred_track in enumerate(pred_tracks):
        pred_by_class[pred_track["category_id"]].append((pred_idx, pred_track))

    for category_id in sorted(set(gt_by_class) | set(pred_by_class)):
        curr_gt = gt_by_class.get(category_id, [])
        curr_pred = pred_by_class.get(category_id, [])
        if not curr_gt or not curr_pred:
            continue

        cost = np.ones((len(curr_gt), len(curr_pred)), dtype=float)
        for gt_row, (_, gt_track) in enumerate(curr_gt):
            denom = max(1, len(gt_track["active_frames"]))
            for pred_col, (_, pred_track) in enumerate(curr_pred):
                common_frames = gt_track["active_frames"] & pred_track["active_frames"]
                if not common_frames:
                    continue
                iou_sum = 0.0
                for frame_idx in common_frames:
                    iou_sum += safe_iou(gt_track["segmentations"][frame_idx], pred_track["segmentations"][frame_idx])
                avg_iou = iou_sum / denom
                cost[gt_row, pred_col] = 1.0 - avg_iou

        row_ind, col_ind = linear_sum_assignment(cost)
        for gt_row, pred_col in zip(row_ind, col_ind):
            avg_iou = 1.0 - cost[gt_row, pred_col]
            if avg_iou <= min_track_iou:
                continue
            pred_idx = curr_pred[pred_col][0]
            gt_idx = curr_gt[gt_row][0]
            matches[pred_idx] = gt_idx
    return matches


def build_mask_oracle(pred_tracks, gt_tracks, video_info_item, min_track_iou):
    output_tracks = [make_output_track(track, track["segmentations"]) for track in pred_tracks]
    matches = build_track_match(gt_tracks, pred_tracks, min_track_iou)
    for pred_idx, gt_idx in matches.items():
        gt_track = gt_tracks[gt_idx]
        pred_track = pred_tracks[pred_idx]
        for frame_idx in range(video_info_item["length"]):
            if not is_active(pred_track["segmentations"][frame_idx]):
                continue
            if not is_active(gt_track["segmentations"][frame_idx]):
                continue
            output_tracks[pred_idx]["segmentations"][frame_idx] = copy_segmentation(gt_track["segmentations"][frame_idx])
    return output_tracks


def build_sounding_oracle(pred_tracks, gt_tracks, video_info_item, min_track_iou):
    empty_seg = encode_empty_rle(video_info_item["height"], video_info_item["width"])
    output_tracks = [make_output_track(track, track["segmentations"]) for track in pred_tracks]
    matches = build_track_match(gt_tracks, pred_tracks, min_track_iou)
    for pred_idx, gt_idx in matches.items():
        gt_track = gt_tracks[gt_idx]
        pred_track = pred_tracks[pred_idx]
        for frame_idx in range(video_info_item["length"]):
            gt_seg = gt_track["segmentations"][frame_idx]
            pred_seg = pred_track["segmentations"][frame_idx]
            if is_active(gt_seg):
                if is_active(pred_seg):
                    output_tracks[pred_idx]["segmentations"][frame_idx] = copy_segmentation(pred_seg)
                else:
                    output_tracks[pred_idx]["segmentations"][frame_idx] = copy_segmentation(gt_seg)
            else:
                output_tracks[pred_idx]["segmentations"][frame_idx] = copy_segmentation(empty_seg)
    return output_tracks


def build_association_oracle(pred_tracks, gt_tracks, video_info_item, min_frame_iou):
    empty_seg = encode_empty_rle(video_info_item["height"], video_info_item["width"])
    used_frames = [set() for _ in pred_tracks]
    oracle_tracks = {}
    oracle_scores = defaultdict(list)

    gt_by_class = defaultdict(list)
    pred_by_class = defaultdict(list)
    for gt_idx, gt_track in enumerate(gt_tracks):
        gt_by_class[gt_track["category_id"]].append((gt_idx, gt_track))
    for pred_idx, pred_track in enumerate(pred_tracks):
        pred_by_class[pred_track["category_id"]].append((pred_idx, pred_track))

    for category_id in sorted(set(gt_by_class) | set(pred_by_class)):
        class_gt = gt_by_class.get(category_id, [])
        class_pred = pred_by_class.get(category_id, [])
        if not class_gt or not class_pred:
            continue

        for frame_idx in range(video_info_item["length"]):
            gt_frame = [(gt_idx, gt_track["segmentations"][frame_idx]) for gt_idx, gt_track in class_gt if is_active(gt_track["segmentations"][frame_idx])]
            pred_frame = [(pred_idx, pred_track["segmentations"][frame_idx]) for pred_idx, pred_track in class_pred if is_active(pred_track["segmentations"][frame_idx])]
            if not gt_frame or not pred_frame:
                continue

            gt_masks = [seg for _, seg in gt_frame]
            pred_masks = [seg for _, seg in pred_frame]
            iou_matrix = mask_utils.iou(gt_masks, pred_masks, [False] * len(pred_masks))
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            for gt_row, pred_col in zip(row_ind, col_ind):
                iou = float(iou_matrix[gt_row, pred_col])
                if iou <= min_frame_iou:
                    continue
                gt_idx = gt_frame[gt_row][0]
                pred_idx = pred_frame[pred_col][0]
                if gt_idx not in oracle_tracks:
                    oracle_tracks[gt_idx] = {
                        "video_id": gt_tracks[gt_idx]["video_id"],
                        "score": 0.0,
                        "category_id": gt_tracks[gt_idx]["category_id"],
                        "segmentations": [copy_segmentation(empty_seg) for _ in range(video_info_item["length"])],
                    }
                oracle_tracks[gt_idx]["segmentations"][frame_idx] = copy_segmentation(pred_tracks[pred_idx]["segmentations"][frame_idx])
                oracle_scores[gt_idx].append(pred_tracks[pred_idx]["score"])
                used_frames[pred_idx].add(frame_idx)

    output_tracks = []
    for gt_idx in sorted(oracle_tracks):
        oracle_tracks[gt_idx]["score"] = float(max(oracle_scores[gt_idx]))
        output_tracks.append(oracle_tracks[gt_idx])

    for pred_idx, pred_track in enumerate(pred_tracks):
        segmentations = []
        has_active = False
        for frame_idx, seg in enumerate(pred_track["segmentations"]):
            if frame_idx in used_frames[pred_idx] and is_active(seg):
                segmentations.append(copy_segmentation(empty_seg))
            else:
                segmentations.append(copy_segmentation(seg))
                if is_active(seg):
                    has_active = True
        if has_active:
            output_tracks.append(
                {
                    "video_id": pred_track["video_id"],
                    "score": float(pred_track["score"]),
                    "category_id": int(pred_track["category_id"]),
                    "segmentations": segmentations,
                }
            )
    return output_tracks


def build_full_oracle(gt_tracks, video_info_item):
    return [make_gt_output_track(gt_track, video_info_item, score=1.0) for gt_track in gt_tracks]


def write_tracker(output_dir, tracker_name, predictions):
    tracker_dir = Path(output_dir) / tracker_name
    tracker_dir.mkdir(parents=True, exist_ok=True)
    with open(tracker_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f)


def evaluate_trackers(trackers_folder, gt_file, trackers_to_eval):
    freeze_config = aviseval.Evaluator.get_default_eval_config()
    dataset_config = aviseval.datasets.AVIS.get_default_dataset_config()
    dataset_config["TRACKERS_FOLDER"] = str(trackers_folder)
    dataset_config["GT_File"] = Path(gt_file).name
    dataset_config["GT_FOLDER"] = str(Path(gt_file).resolve().parent)
    dataset_config["TRACKERS_TO_EVAL"] = trackers_to_eval
    metrics_config = {"METRICS": ["TrackMAP", "HOTA"]}

    evaluator = aviseval.Evaluator({**freeze_config, **dataset_config, **metrics_config})
    dataset_list = [aviseval.datasets.AVIS(dataset_config)]
    metrics_list = []
    for metric in [aviseval.metrics.TrackMAP, aviseval.metrics.HOTA]:
        if metric.get_name() not in metrics_config["METRICS"]:
            continue
        if metric == aviseval.metrics.TrackMAP:
            metric_config = metric.get_default_metric_config()
            metric_config["USE_TIME_RANGES"] = False
            metric_config["AREA_RANGES"] = [[0 ** 2, 128 ** 2], [128 ** 2, 256 ** 2], [256 ** 2, 1e5 ** 2]]
            metrics_list.append(metric(metric_config))
        else:
            metrics_list.append(metric())
    output_res, _ = evaluator.evaluate(dataset_list, metrics_list)
    return output_res["AVIS"]


def summarize_metrics(name, metrics):
    return {
        "tracker": name,
        "AP_all": metrics["AP_all"],
        "HOTA": metrics["HOTA"],
        "DetA": metrics["DetA"],
        "AssA": metrics["AssA"],
        "LocA": metrics["LocA"],
        "FA": metrics["FA"],
        "FAn": metrics["FAn"],
        "FAs": metrics["FAs"],
        "FAm": metrics["FAm"],
    }


def print_summary(summary):
    print("\nOracle summary")
    print("-" * 88)
    print(f"{'tracker':<16}{'AP_all':>10}{'HOTA':>10}{'DetA':>10}{'AssA':>10}{'LocA':>10}{'FA':>10}")
    print("-" * 88)
    for item in summary:
        print(
            f"{item['tracker']:<16}"
            f"{item['AP_all']:>10.2f}"
            f"{item['HOTA']:>10.2f}"
            f"{item['DetA']:>10.2f}"
            f"{item['AssA']:>10.2f}"
            f"{item['LocA']:>10.2f}"
            f"{item['FA']:>10.2f}"
        )
    print("-" * 88)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_data, video_info, gt_tracks_by_video = load_gt(args.gt)
    _, pred_tracks_by_video = load_predictions(args.pred, video_info)

    if "baseline" in args.modes:
        baseline_dir = out_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.pred, baseline_dir / "results.json")

    mode_builders = {
        "mask": lambda pred_tracks, gt_tracks, info: build_mask_oracle(pred_tracks, gt_tracks, info, args.min_track_iou),
        "association": lambda pred_tracks, gt_tracks, info: build_association_oracle(pred_tracks, gt_tracks, info, args.min_frame_iou),
        "sounding": lambda pred_tracks, gt_tracks, info: build_sounding_oracle(pred_tracks, gt_tracks, info, args.min_track_iou),
        "full": lambda pred_tracks, gt_tracks, info: build_full_oracle(gt_tracks, info),
    }

    for mode in args.modes:
        if mode == "baseline":
            continue
        all_predictions = []
        for video_id, info in video_info.items():
            gt_tracks = gt_tracks_by_video.get(video_id, [])
            pred_tracks = pred_tracks_by_video.get(video_id, [])
            tracker_predictions = mode_builders[mode](pred_tracks, gt_tracks, info)
            all_predictions.extend(tracker_predictions)
        write_tracker(out_dir, mode, all_predictions)

    trackers_to_eval = [mode for mode in args.modes if (out_dir / mode).exists()]
    evaluation = evaluate_trackers(out_dir, args.gt, trackers_to_eval)

    summary = [summarize_metrics(name, evaluation[name]) for name in trackers_to_eval]
    print_summary(summary)

    with open(out_dir / "oracle_summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "raw": evaluation}, f, indent=2)

    with open(out_dir / "oracle_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "gt": str(Path(args.gt).resolve()),
                "pred": str(Path(args.pred).resolve()),
                "modes": args.modes,
                "min_track_iou": args.min_track_iou,
                "min_frame_iou": args.min_frame_iou,
                "gt_categories": len(gt_data["categories"]),
                "gt_videos": len(gt_data["videos"]),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
