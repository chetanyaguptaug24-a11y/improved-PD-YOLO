import argparse
import csv
import json
import time
import tempfile
import yaml
from pathlib import Path

import torch

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and compare PD-YOLO models")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/Bump.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="runs/experiments/evaluation")
    parser.add_argument("models", nargs="+", help="name=checkpoint-or-yaml paths")
    return parser.parse_args()


def resolve_dataset_yaml(dataset):
    root = Path(__file__).resolve().parent
    source = Path(dataset)
    if not source.is_absolute():
        source = root / source
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    voc_root = root / "dataset" / "VOC2007"
    if voc_root.is_dir():
        data["path"] = str(voc_root)
        for split in ("train", "val", "test"):
            if data.get(split):
                data[split] = str(voc_root / data[split])
    resolved = Path(tempfile.gettempdir()) / f"improved_pd_yolo_eval_{source.stem}.yaml"
    resolved.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(resolved)


def main():
    args = parse_args()
    args.data = resolve_dataset_yaml(args.data)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in args.models:
        name, path = item.split("=", 1)
        model = YOLO(path)
        metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
                            device=args.device, plots=False, verbose=False)
        test_dir = Path(__file__).resolve().parent / "dataset" / "VOC2007" / "test" / "images"
        test_images = [p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        start = time.perf_counter()
        with torch.inference_mode():
            model.predict(source=str(test_dir), imgsz=args.imgsz, device=args.device,
                          batch=args.batch, verbose=False)
        elapsed = time.perf_counter() - start
        latency_ms = 1000 * elapsed / max(len(test_images), 1)
        fps = 1000 / max(latency_ms, 1e-9)
        parameters = sum(p.numel() for p in model.model.parameters())
        try:
            gflops = model.info(verbose=False)[1]
        except Exception:
            gflops = 0.0
        row = {
            "Model": name,
            "Precision": float(metrics.box.mp),
            "Recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "Parameters": parameters,
            "GFLOPs": gflops,
            "Inference latency (ms/image)": latency_ms,
            "FPS": fps,
            "Model size (bytes)": Path(path).stat().st_size if Path(path).is_file() else 0,
        }
        rows.append(row)

    csv_path = output / "evaluation.csv"
    json_path = output / "evaluation.json"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
