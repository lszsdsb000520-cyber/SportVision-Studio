"""本地复算测试集分数（不需要联网）。

与 tools/evaluate_holdout.py 的指标口径完全一致：
- 数据：datasets/holdout25/*.mp4 + 同名 csv（BlurBall 官方 test match 25 的前 6 个 clip）
- 口径：把预测像素换算到 512x288 空间，与真值距离 <= 6px 记 TP
- 输出：TP/FP/FN/TN、precision/recall/F1、可见帧平均定位误差

用法：
    .venv-app/Scripts/python.exe tools/eval_holdout_local.py
    .venv-app/Scripts/python.exe tools/eval_holdout_local.py --folder datasets/holdout25
"""
from pathlib import Path
import argparse
import csv
import json
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pingpong.detector import analyze_video


def evaluate_clip(video: Path, model: Path, threshold=0.5):
    labels = list(csv.DictReader(video.with_suffix(".csv").open(encoding="utf-8")))
    project = analyze_video(video, model, threshold=threshold)
    if project["frame_count"] != len(labels):
        raise RuntimeError(f"帧数不一致 {video.name}: {project['frame_count']} vs {len(labels)}")
    w, h = project["width"], project["height"]
    counts = dict(tp=0, fp=0, fn=0, tn=0)
    errors = []
    for p, label in zip(project["tracks"], labels):
        if p["frame"] != int(label["Frame"]):
            raise RuntimeError(f"帧号错位 {video.name}@{p['frame']}")
        visible = label["Visibility"] == "1"
        found = p["x"] is not None
        if visible and found:
            err = float(np.hypot((p["x"] - float(label["X"])) * 512 / w,
                                 (p["y"] - float(label["Y"])) * 288 / h))
            errors.append(err)
            if err <= 6:
                counts["tp"] += 1
            else:
                counts["fp"] += 1
                counts["fn"] += 1
        elif visible:
            counts["fn"] += 1
        elif found:
            counts["fp"] += 1
        else:
            counts["tn"] += 1
    return dict(file=video.name, frames=len(labels), **counts,
                mean_error_512px=float(np.mean(errors)) if errors else None,
                performance=project["performance"])


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, default=root / "datasets/holdout25")
    ap.add_argument("--model", type=Path, default=root / "models/detector.onnx")
    ap.add_argument("--output", type=Path, default=root / "reports/holdout_evaluation_local.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    videos = sorted(args.folder.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"没有找到测试视频：{args.folder}")

    clips, tp = [], 0
    fp = fn = tn = frames = 0
    all_err = []
    t0 = time.perf_counter()
    for video in videos:
        r = evaluate_clip(video, args.model,args.threshold)
        tp += r["tp"]; fp += r["fp"]; fn += r["fn"]; tn += r["tn"]; frames += r["frames"]
        if r["mean_error_512px"] is not None:
            all_err.append(r["mean_error_512px"])
        clips.append(r)
        fps = r["performance"].get("inference_fps")
        print(f"{r['file']}: frames={r['frames']} tp={r['tp']} fp={r['fp']} fn={r['fn']} "
              f"tn={r['tn']} err={r['mean_error_512px']} infer={fps and round(fps, 1)}", flush=True)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    result = dict(
        dataset="BlurBall official test match 25（本地 datasets/holdout25，前 6 个 clip）",
        model=str(args.model), clips=clips, frames=frames,
        tp=tp, fp=fp, fn=fn, tn=tn, threshold=args.threshold, matching_radius_512px=6,
        precision=precision, recall=recall,
        f1=2 * precision * recall / max(1e-9, precision + recall),
        mean_error_on_detected_visible_frames_512px=float(np.mean(all_err)) if all_err else None,
        elapsed_seconds=round(time.perf_counter() - t0, 2),
        caveats=["这是球检测（逐帧球心）分数，测试集不含触台/出界事件真值。",
                 "小样本子集（6 个 clip），不等于官方全量或比赛成绩。"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "clips"}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
