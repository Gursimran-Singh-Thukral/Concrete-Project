"""SDNET2018 wall classification and approximate sliding-window localization."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import threading
import time

from depth_sensor import AIR_SPEED, scale_boxes

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo"))
DEFAULT_SOURCE = ROOT / "DATA_Maguire_20180517_ALL/DATA_Maguire_20180517_ALL/SDNET2018"
CLASSES = ("crack", "no_crack")


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def prepare(args):
    from PIL import Image
    from collections import Counter
    source, output = Path(args.source).resolve(), Path(args.dataset).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Dataset output is nonempty: {output}. Use a new --dataset directory.")
    rows, hashes, groups, bad = [], {}, {}, []
    excluded_hashes, duplicate_records = set(), []
    for surface in args.surfaces:
        for prefix, label in (("C", "crack"), ("U", "no_crack")):
            folder = source / surface / (prefix + surface)
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                match = re.fullmatch(r"(\d+)-(\d+)(?:_\d+)?", path.stem)
                if not match:
                    raise ValueError(f"Unrecognized source-photo ID: {path.name}")
                try:
                    with Image.open(path) as image:
                        image.load()
                        if image.size != (256, 256) or image.mode != "RGB":
                            raise ValueError(f"Unexpected dimensions/mode: {image.size}, {image.mode}")
                except Exception as error:
                    bad.append({"path": str(path), "error": str(error)})
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                group = surface + ":" + match[1]
                row = {"source": str(path), "label": label, "group": group, "sha256": digest}
                if digest in hashes:
                    previous = hashes[digest]
                    reason = "same_group_same_label"
                    if previous["label"] != label:
                        reason = "conflicting_labels_exclude_all_copies"
                        excluded_hashes.add(digest)
                    elif previous["group"] != group:
                        reason = "cross_group_duplicate_exclude_all_copies"
                        excluded_hashes.add(digest)
                    duplicate_records.append({"first": previous, "duplicate": row, "reason": reason})
                    continue
                hashes[digest] = row
                rows.append(row)
    rows = [row for row in rows if row["sha256"] not in excluded_hashes]
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    if bad:
        save_json(ROOT / "reports/corrupt_images.json", bad)
        raise ValueError(f"Found {len(bad)} invalid images; see reports/corrupt_images.json.")
    if len(groups) < 10:
        raise ValueError("At least 10 source photographs are required for a grouped split.")
    # Select a reproducible split by counts only, before training or measuring performance.
    ids = sorted(groups)
    rng = random.Random(args.seed)
    n_val = max(1, round(len(ids) * 0.15))
    n_test = max(1, round(len(ids) * 0.15))
    totals = Counter(row["label"] for row in rows)
    best = None
    for _ in range(2000):
        shuffled = rng.sample(ids, len(ids))
        splits = {"val": shuffled[:n_val], "test": shuffled[n_val:n_val+n_test], "train": shuffled[n_val+n_test:]}
        score = 0.0
        valid = True
        for name, selected in splits.items():
            count = Counter(row["label"] for group in selected for row in groups[group])
            fraction = len(selected) / len(ids)
            if any(count[label] == 0 for label in CLASSES):
                valid = False
                break
            score += sum((count[label] / totals[label] - fraction) ** 2 for label in CLASSES)
        if valid and (best is None or score < best[0]):
            best = score, splits
    if best is None:
        raise ValueError("Could not produce splits containing both classes.")
    assignment = {group: split for split, selected in best[1].items() for group in selected}
    for split in ("train", "val", "test"):
        for label in CLASSES:
            (output / split / label).mkdir(parents=True, exist_ok=True)
    links, copies = 0, 0
    for row in rows:
        row["split"] = assignment[row["group"]]
        src = Path(row["source"])
        target = output / row["split"] / row["label"] / (row["group"].split(":")[0] + "_" + src.name)
        try:
            os.link(src, target)
            links += 1
        except OSError:
            shutil.copy2(src, target)
            copies += 1
        row["prepared"] = str(target)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source": str(source), "surfaces": args.surfaces, "seed": args.seed,
        "split_unit": "surface + filename prefix before hyphen (inferred source photograph)",
        "image_count": len(rows), "source_photographs": len(groups),
        "hardlinks": links, "copies": copies, "invalid_images": len(bad),
        "duplicate_copies_removed": len(duplicate_records),
        "ambiguous_unique_images_excluded": len(excluded_hashes),
        "manifest_sha256": hashlib.sha256((output / "manifest.csv").read_bytes()).hexdigest(),
        "splits": {split: {"source_photographs": len(selected), "groups": sorted(selected),
                            "classes": dict(Counter(row["label"] for row in rows if row["split"] == split))}
                   for split, selected in best[1].items()},
    }
    save_json(output / "summary.json", summary)
    save_json(output / "duplicate_audit.json", duplicate_records)
    print(json.dumps(summary, indent=2))


def device_value(value):
    if value != "auto":
        return value
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


def train(args):
    import torch
    from ultralytics import YOLO
    from crack_training import FullPatchTrainer
    dataset = Path(args.dataset).resolve()
    if not (dataset / "manifest.csv").exists():
        raise ValueError("Run prepare first to generate audited, grouped splits.")
    model = YOLO(args.initial)
    start = time.perf_counter()
    model.train(
        trainer=FullPatchTrainer, data=str(dataset), epochs=args.epochs,
        batch=args.batch, imgsz=224, device=device_value(args.device),
        workers=args.workers, project=str(ROOT / "runs"), name=args.name,
        seed=args.seed, deterministic=True, optimizer="AdamW", lr0=0.001,
        patience=args.epochs, cache=False, amp=True, plots=True,
        auto_augment=None, erasing=0.0,
    )
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights/best.pt"
    if not best.is_file():
        raise RuntimeError("Training did not produce best.pt.")
    destination = ROOT / "models"
    destination.mkdir(exist_ok=True)
    shutil.copy2(best, destination / "best.pt")
    shutil.copy2(run_dir / "weights/last.pt", destination / "last.pt")
    save_json(destination / "training.json", {
        "initial_weights": args.initial, "dataset": str(dataset), "epochs_requested": args.epochs,
        "epochs_completed": model.trainer.epoch + 1, "batch": args.batch, "imgsz": 224,
        "seed": args.seed, "device": device_value(args.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "run_dir": str(run_dir), "training_seconds": time.perf_counter() - start,
        "preprocessing": "Resize entire RGB patch to 224x224; float [0,1]",
        "augmentation": "Horizontal/vertical flips, mild brightness/contrast/saturation/hue jitter",
        "selection": "Best validation top-1 checkpoint; threshold tuned on validation F2",
    })
    args.weights = str(destination / "best.pt")
    evaluate(args)


def load_model(weights):
    from ultralytics import YOLO
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(f"Trained checkpoint missing: {path}")
    model = YOLO(str(path))
    if model.task != "classify" or set(model.names.values()) != set(CLASSES):
        raise ValueError(f"Expected binary crack classification weights, got {model.names}")
    return model


def scores(model, images, device, batch=24):
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    import torchvision.transforms as T
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    crack_index = next(index for index, name in model.names.items() if name == "crack")
    output = []
    for start in range(0, len(images), batch):
        # Tensor inputs bypass library crop transforms and exactly match full-patch validation.
        tensors = torch.stack([transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                               for img in images[start:start+batch]])
        results = model.predict(tensors, imgsz=224, device=device, verbose=False)
        output.extend(float(result.probs.data[crack_index].cpu()) for result in results)
    return np.asarray(output)


def binary_metrics(labels, probabilities, threshold):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, average_precision_score, roc_auc_score
    predicted = probabilities >= threshold
    return {
        "threshold": float(threshold), "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "f2": float(fbeta_score(labels, predicted, beta=2, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": ["no_crack", "crack"],
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "samples": len(labels),
    }


def evaluate(args):
    import cv2
    import numpy as np
    from sklearn.metrics import precision_recall_curve
    model = load_model(args.weights)
    dataset = Path(args.dataset).resolve()
    output = ROOT / "models"
    output.mkdir(exist_ok=True)
    with (dataset / "manifest.csv").open(encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    group_sets = {s: {r["group"] for r in manifest if r["split"] == s} for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if group_sets[a] & group_sets[b]:
            raise ValueError("Source group leakage in manifest.")
    device = device_value(args.device)
    warmup = cv2.imread(next(row["prepared"] for row in manifest if row["split"] == "val"))
    scores(model, [warmup], device, args.batch)
    report = {"weights": str(Path(args.weights).resolve()), "weights_sha256": hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
              "dataset_manifest_sha256": hashlib.sha256((dataset / "manifest.csv").read_bytes()).hexdigest(),
              "positive_class": "crack", "threshold_selection": "Maximum validation F2 (recall-weighted), before test evaluation",
              "scope": "Dry SDNET2018 photographs; not validated underwater or for dam safety", "splits": {}}
    threshold = None
    for split in ("val", "test"):
        selected = [row for row in manifest if row["split"] == split]
        labels = np.asarray([int(row["label"] == "crack") for row in selected])
        predicted, elapsed = [], 0.0
        for start in range(0, len(selected), args.batch):
            batch_rows = selected[start:start+args.batch]
            images = [cv2.imread(row["prepared"]) for row in batch_rows]
            if any(image is None for image in images):
                raise ValueError("Unreadable prepared image.")
            tick = time.perf_counter()
            predicted.extend(scores(model, images, device, args.batch))
            elapsed += time.perf_counter() - tick
        probabilities = np.asarray(predicted)
        if split == "val":
            precision, recall, thresholds = precision_recall_curve(labels, probabilities)
            f2 = 5 * precision[:-1] * recall[:-1] / np.maximum(4 * precision[:-1] + recall[:-1], 1e-12)
            threshold = float(thresholds[int(np.argmax(f2))])
        report["splits"][split] = {
            "at_default_threshold": binary_metrics(labels, probabilities, 0.5),
            "at_selected_threshold": binary_metrics(labels, probabilities, threshold),
            "source_photographs": len(group_sets[split]),
            "inference_ms_per_patch": elapsed / len(selected) * 1000,
            "inference_timing": "Batch throughput including resize and prediction, excluding file decode; warmup excluded",
        }
        with (output / f"{split}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "group", "true_label", "crack_probability", "predicted_label"])
            for row, prob in zip(selected, probabilities):
                writer.writerow([row["source"], row["group"], row["label"], float(prob), "crack" if prob >= threshold else "no_crack"])
        print(split, json.dumps(report["splits"][split], indent=2))
    report["selected_threshold"] = threshold
    save_json(output / "metrics.json", report)
    plot_results(report, output)
    write_results_report(report, dataset)


def write_results_report(report, dataset):
    summary = json.loads((dataset / "summary.json").read_text(encoding="utf-8"))
    training_path = ROOT / "models/training.json"
    training = json.loads(training_path.read_text(encoding="utf-8")) if training_path.exists() else {}
    lines = [
        "# Measured crack-classification results", "",
        "This experiment uses dry SDNET2018 concrete wall images. It provides a preliminary",
        "crack classifier and an approximate candidate-region display for the dam inspection project.", "",
        f"- Checkpoint: `{report['weights']}`",
        f"- Checkpoint SHA-256: `{report['weights_sha256']}`",
        f"- Completed training epochs: {training.get('epochs_completed', 'not recorded')}",
        f"- GPU: {training.get('gpu', 'not recorded')}",
        f"- Usable images: {summary['image_count']}; source photographs: {summary['source_photographs']}.",
        f"- Duplicate copies removed: {summary['duplicate_copies_removed']}; ambiguous unique images excluded: {summary['ambiguous_unique_images_excluded']}.",
        "- Split by inferred original photograph; no source group or exact hash crosses splits.",
        f"- Selected threshold: **{report['selected_threshold']:.6f}**, maximizing validation F2.", "",
        "| Split | Threshold | Accuracy | Crack precision | Crack recall | F1 | F2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("val", "test"):
        for setting in ("at_default_threshold", "at_selected_threshold"):
            item = report["splits"][split][setting]
            lines.append(f"| {split} | {item['threshold']:.6f} | {item['accuracy']:.2%} | {item['precision']:.2%} | {item['recall']:.2%} | {item['f1']:.4f} | {item['f2']:.4f} |")
    test = report["splits"]["test"]["at_selected_threshold"]
    tn, fp = test["confusion_matrix"][0]
    fn, tp = test["confusion_matrix"][1]
    lines += ["", "Test confusion matrix at the selected threshold:", "",
              "| Actual / Predicted | No crack | Crack |", "|---|---:|---:|",
              f"| No crack | {tn} | {fp} |", f"| Crack | {fn} | {tp} |", "",
              f"The held-out test has {test['samples']} patches from {report['splits']['test']['source_photographs']} source photographs.",
              f"An always-no-crack baseline would have {(tn + fp) / test['samples']:.2%} accuracy and zero crack recall.",
              f"Test average precision: {test['average_precision']:.4f}; ROC AUC: {test['roc_auc']:.4f}.",
              f"Measured batch inference throughput: {report['splits']['test']['inference_ms_per_patch']:.3f} ms per patch, excluding decode and warmup.", "",
              "![Test evaluation](models/evaluation.png)", "",
              "## Interpretation", "",
              "The test set was not used to tune the model or threshold. Photograph grouping reduces patch leakage,",
              "but the wall photographs come from the same dataset and do not validate new dams or underwater conditions.",
              "Crack scores are not independently calibrated probabilities. Sliding-window boxes have no localization",
              "ground truth in this dataset and cannot be scored as precise crack detections or used as a crack count.", "",
              "The image and video commands implement inspection software; a physical camera, freshwater specimens,",
              "lighting, distance, turbidity and motion still need experimental validation.", "",
              "Full measurements and individual predictions are in `models/metrics.json`,",
              "`models/val_predictions.csv` and `models/test_predictions.csv`. See `README.md` for commands.", ""]
    (ROOT / "TRAINING_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def plot_results(report, output):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    matrix = report["splits"]["test"]["at_selected_threshold"]["confusion_matrix"]
    ConfusionMatrixDisplay(np.asarray(matrix), display_labels=["No crack", "Crack"]).plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Held-out source photographs")
    with (output / "test_predictions.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    precision, recall, _ = precision_recall_curve([r["true_label"] == "crack" for r in rows], [float(r["crack_probability"]) for r in rows])
    axes[1].plot(recall, precision)
    axes[1].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title="Test precision–recall curve")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "evaluation.png", dpi=180)
    plt.close(fig)


def selected_threshold(args):
    if args.threshold is not None:
        return args.threshold
    metadata = Path(args.weights).parent / "metrics.json"
    if not metadata.exists():
        raise ValueError("No tuned threshold found; run evaluate or pass --threshold.")
    report = json.loads(metadata.read_text(encoding="utf-8"))
    digest = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
    if digest != report["weights_sha256"]:
        raise ValueError("Threshold metadata belongs to different weights; run evaluate again.")
    return report["selected_threshold"]


def draw_candidates(annotated, boxes):
    import cv2
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        cv2.rectangle(annotated, (x, y), (x+w-1, y+h-1), (0, 180, 255), 2)
        cv2.putText(annotated, f"Candidate {box['mean_patch_score']:.2f}", (x, max(18, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
    cv2.putText(annotated, "Approximate crack candidates", (8, annotated.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1)


class DetectionWorker:
    """Runs candidate_regions on the newest submitted frame in a background thread."""

    def __init__(self, model, args, threshold):
        self.model, self.args, self.threshold = model, args, threshold
        self.lock = threading.Lock()
        self.pending, self.result, self.count = None, None, 0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self.lock:
            self.pending = frame

    def latest(self):
        with self.lock:
            return self.count, self.result

    def run(self):
        while not self.stopped.is_set():
            with self.lock:
                frame, self.pending = self.pending, None
            if frame is None:
                time.sleep(0.005)
                continue
            tick = time.perf_counter()
            _, _, metadata = candidate_regions(frame, self.model, self.args, self.threshold)
            metadata.update(processing_seconds=time.perf_counter() - tick,
                            frame_width=frame.shape[1], frame_height=frame.shape[0])
            with self.lock:
                self.result, self.count = metadata, self.count + 1

    def stop(self):
        self.stopped.set()
        self.thread.join()


def candidate_regions(frame, model, args, threshold):
    import cv2
    import numpy as np
    height, width = frame.shape[:2]
    patch = min(args.patch, height, width)
    stride = min(args.stride, patch)
    def positions(length):
        return sorted(set(list(range(0, length - patch + 1, stride)) + [length - patch]))
    coords = [(x, y) for y in positions(height) for x in positions(width)]
    tiles = [frame[y:y+patch, x:x+patch] for x, y in coords]
    probabilities = scores(model, tiles, device_value(args.device), args.batch)
    heat = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    for (x, y), probability in zip(coords, probabilities):
        heat[y:y+patch, x:x+patch] += probability
        count[y:y+patch, x:x+patch] += 1
    heat /= np.maximum(count, 1)
    mask = (heat >= threshold).astype(np.uint8)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    annotated, boxes = frame.copy(), []
    for label in range(1, components):
        x, y, w, h, area = stats[label].tolist()
        if area < args.min_area:
            continue
        confidence = float(heat[labels == label].mean())
        boxes.append({"x": x, "y": y, "width": w, "height": h, "mean_patch_score": confidence})
    draw_candidates(annotated, boxes)
    return annotated, heat, {"threshold": threshold, "patch": patch, "stride": stride,
                             "tiles": len(coords), "boxes": boxes,
                             "interpretation": "Coarse patch candidates, not crack geometry or physical crack count"}


def image_command(args):
    import cv2
    import numpy as np
    frame = cv2.imread(args.source)
    if frame is None:
        raise ValueError(f"Cannot read image: {args.source}")
    model = load_model(args.weights)
    threshold = selected_threshold(args)
    start = time.perf_counter()
    annotated, heat, metadata = candidate_regions(frame, model, args, threshold)
    metadata["processing_seconds"] = time.perf_counter() - start
    metadata["source"] = str(Path(args.source).resolve())
    metadata["weights"] = str(Path(args.weights).resolve())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), annotated):
        raise OSError(f"Cannot save {output}")
    heat_path = output.with_name(output.stem + "_heatmap.png")
    cv2.imwrite(str(heat_path), cv2.applyColorMap((heat * 255).clip(0, 255).astype(np.uint8), cv2.COLORMAP_TURBO))
    save_json(output.with_suffix(".json"), metadata)
    print(json.dumps(metadata, indent=2))


class PiCameraCapture:
    """Minimal cv2.VideoCapture look-alike for the Raspberry Pi Camera Module."""

    def __init__(self, size=(1280, 720)):
        from picamera2 import Picamera2
        self.camera = Picamera2()
        # "RGB888" yields BGR-ordered arrays, which is what OpenCV expects.
        self.camera.configure(self.camera.create_video_configuration(main={"size": size, "format": "RGB888"}))
        self.camera.start()
        self.started = time.perf_counter()

    def isOpened(self):
        return True

    def read(self):
        return True, self.camera.capture_array()

    def get(self, prop):
        import cv2
        if prop == cv2.CAP_PROP_POS_MSEC:
            return (time.perf_counter() - self.started) * 1000
        return 0

    def release(self):
        self.camera.stop()
        self.camera.close()


def live_command(args):
    import cv2
    model = load_model(args.weights)
    threshold = selected_threshold(args)
    if args.source == "picamera":
        capture = PiCameraCapture()
    else:
        source = int(args.source) if args.source.isdecimal() else args.source
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise ValueError(f"Cannot open camera/video: {args.source}")
    writer, frame_index = None, 0
    sensor = None
    if args.ultrasonic:
        from depth_sensor import UltrasonicSensor
        sensor = UltrasonicSensor(args.trig, args.echo, args.sound_speed, args.max_range)
    worker, logged = DetectionWorker(model, args, threshold) if args.smooth else None, 0
    log = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        log = Path(args.output).with_suffix(".jsonl").open("w", encoding="utf-8")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if args.max_width and frame.shape[1] > args.max_width:
                scale = args.max_width / frame.shape[1]
                frame = cv2.resize(frame, (args.max_width, round(frame.shape[0] * scale)))
            distance = sensor.read() if sensor else None
            if worker:
                worker.submit(frame)
                detections, metadata = worker.latest()
                annotated = frame.copy()
                if metadata:
                    draw_candidates(annotated, metadata["boxes"])
                if metadata and detections != logged:
                    logged = detections
                    metadata = dict(metadata, frame=frame_index, video_time_ms=capture.get(cv2.CAP_PROP_POS_MSEC))
                    if sensor:
                        metadata = scale_boxes(metadata, distance, args.hfov)
                    if log:
                        log.write(json.dumps(metadata) + "\n")
            else:
                tick = time.perf_counter()
                annotated, _, metadata = candidate_regions(frame, model, args, threshold)
                metadata.update(frame=frame_index, processing_seconds=time.perf_counter() - tick,
                                video_time_ms=capture.get(cv2.CAP_PROP_POS_MSEC),
                                frame_width=frame.shape[1], frame_height=frame.shape[0])
                if sensor:
                    metadata = scale_boxes(metadata, distance, args.hfov)
                if log:
                    log.write(json.dumps(metadata) + "\n")
            if sensor:
                label = f"Wall distance: {distance:.2f} m" if distance is not None else "Wall distance: out of range"
                cv2.putText(annotated, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            if args.output:
                if writer is None:
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    if not 0 < fps < 240:
                        fps = 25
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise OSError("Cannot open output video writer.")
                writer.write(annotated)
            frame_index += 1
            if not args.no_display:
                cv2.imshow("Dam inspection - crack candidates (Q to exit)", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        if worker:
            worker.stop()
        if sensor:
            sensor.close()
        capture.release()
        if writer:
            writer.release()
        if log:
            log.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    print(f"Processed {frame_index} frames.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Audit images and create source-photo grouped splits")
    p.add_argument("--source", default=str(DEFAULT_SOURCE))
    p.add_argument("--dataset", default=str(ROOT / "dataset"))
    p.add_argument("--surfaces", nargs="+", choices=["W", "D", "P"], default=["W"])
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=prepare)
    for command in ("train", "evaluate", "image", "live"):
        p = sub.add_parser(command)
        p.add_argument("--device", default="auto")
        p.add_argument("--batch", type=int, default=24)
        p.add_argument("--weights", default=str(ROOT / "models/best.pt"))
        if command in ("train", "evaluate"):
            p.add_argument("--dataset", default=str(ROOT / "dataset"))
        if command == "train":
            p.add_argument("--initial", default="yolo11n-cls.pt")
            p.add_argument("--epochs", type=int, default=20)
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--name", default="wall_yolo11n")
        if command in ("image", "live"):
            p.add_argument("--source", required=True)
            p.add_argument("--output", default="outputs/result.jpg" if command == "image" else None)
            p.add_argument("--threshold", type=float)
            p.add_argument("--patch", type=int, default=256)
            p.add_argument("--stride", type=int, default=128)
            p.add_argument("--min-area", type=int, default=1024)
        if command == "live":
            p.add_argument("--max-width", type=int, default=640)
            p.add_argument("--no-display", action="store_true")
            p.add_argument("--ultrasonic", action="store_true",
                           help="Read wall distance from a trigger/echo ultrasonic sensor and add it to the overlay and log")
            p.add_argument("--trig", type=int, default=23, help="BCM GPIO number wired to the sensor TRIG pin")
            p.add_argument("--echo", type=int, default=24, help="BCM GPIO number wired to the sensor ECHO pin (via divider)")
            p.add_argument("--sound-speed", type=float, default=AIR_SPEED,
                           help="Speed of sound in the medium in m/s (343 for air, about 1480 for fresh water)")
            p.add_argument("--max-range", type=float, default=2.0, help="Largest distance to report, in metres")
            p.add_argument("--hfov", type=float, default=66.0,
                           help="Horizontal field of view in the medium, in degrees, for converting pixels to mm "
                                "(Camera Module 3: 66 in air; Module 2: 62.2)")
            p.add_argument("--smooth", action="store_true",
                           help="Detect in a background thread so the video stays fluid; boxes update when ready")
            p.add_argument("--max-frames", type=int, default=0)
        p.set_defaults(func={"train": train, "evaluate": evaluate, "image": image_command, "live": live_command}[command])
    args = parser.parse_args()
    if hasattr(args, "batch") and args.batch <= 0:
        parser.error("--batch must be positive")
    if hasattr(args, "patch") and (args.patch <= 0 or args.stride <= 0 or args.min_area < 0):
        parser.error("Patch and stride must be positive; min-area must be nonnegative")
    if getattr(args, "threshold", None) is not None and not 0 <= args.threshold <= 1:
        parser.error("Threshold must be in [0,1]")
    args.func(args)


if __name__ == "__main__":
    main()
