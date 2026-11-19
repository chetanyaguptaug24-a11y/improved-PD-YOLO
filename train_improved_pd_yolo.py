import argparse
import tempfile
import yaml
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train Improved PD-YOLO")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/Bump.yaml")
    parser.add_argument("--model", default="ultralytics/cfg/models/YOLOv8n+C3_DCN+DCSP+attention+P2.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="improved_pd_yolo")
    parser.add_argument("--box-loss-type", choices=("ciou", "siou"), default="siou")
    parser.add_argument("--weather-aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--weather-prob", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def resolve_dataset_yaml(dataset):
    """Resolve the repository dataset without hardcoding a machine-specific path."""
    root = Path(__file__).resolve().parent
    source = Path(dataset)
    if not source.is_absolute():
        source = root / source
    if source.is_file():
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        voc_root = root / "dataset" / "VOC2007"
        if voc_root.is_dir():
            data["path"] = str(voc_root)
            for split in ("train", "val", "test"):
                if data.get(split):
                    data[split] = str(voc_root / data[split])
        source = Path(tempfile.gettempdir()) / f"improved_pd_yolo_{source.stem}.yaml"
        source.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return str(source)


def main():
    args = parse_args()
    args.data = resolve_dataset_yaml(args.data)
    model = YOLO(args.resume if args.resume else args.model)
    print("Active model:", args.model)
    print("Active box loss:", args.box_loss_type)
    print("Weather augmentation:", args.weather_aug, "probability:", args.weather_prob)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        seed=args.seed,
        device=args.device or None,
        resume=bool(args.resume),
        project=args.project,
        name=args.name,
        box_loss_type=args.box_loss_type,
        weather_aug=args.weather_aug,
        weather_prob=args.weather_prob,
        workers=args.workers,
        exist_ok=False,
        plots=True,
    )


if __name__ == "__main__":
    main()
