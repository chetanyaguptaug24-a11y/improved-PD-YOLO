"""Generate plots from the reported results tables."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def rows(filename):
    with (ROOT / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def mark_figure(fig, title):
    fig.suptitle(title, fontweight="bold", fontsize=15)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.93))


def save(fig, filename):
    fig.savefig(ROOT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def dataset_comparison():
    data = rows("dataset_comparison.csv")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    datasets = ["VOC2007", "GRDDC2020"]
    models = ["YOLOv8n", "PD-YOLO", "EPD-YOLO"]
    x = np.arange(len(datasets))
    width = 0.23
    colors = ["#9aa0a6", "#4c78a8", "#e45756"]
    for i, model in enumerate(models):
        values = [float(next(r["map50_percent"] for r in data if r["dataset"] == d and r["model"] == model))
                  for d in datasets]
        bars = ax.bar(x + (i - 1) * width, values, width, label=model, color=colors[i])
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_ylabel("mAP@0.5 (%) as reported")
    ax.set_xticks(x, datasets)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.2)
    mark_figure(fig, "Reported results: dataset-level mAP@0.5 comparison")
    save(fig, "dataset_comparison_reported.png")


def component_ablation():
    data = rows("component_ablation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, dataset in zip(axes, ("VOC2007", "GRDDC2020")):
        subset = [r for r in data if r["dataset"] == dataset]
        x = np.arange(len(subset))
        for key, label, color in (
            ("precision_percent", "Precision", "#4c78a8"),
            ("recall_percent", "Recall", "#f2a541"),
            ("map50_percent", "mAP@0.5", "#e45756"),
        ):
            ax.plot(x, [float(r[key]) for r in subset], marker="o", label=label, color=color)
        ax.set_title(dataset)
        ax.set_xticks(x, [r["configuration"] for r in subset])
        ax.set_ylabel("Percent as reported")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    mark_figure(fig, "Reported results: component-wise ablation")
    save(fig, "component_ablation_reported.png")


def scale_ap():
    data = rows("scale_ap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    categories = [("ap_small_percent", "Small"), ("ap_medium_percent", "Medium"), ("ap_large_percent", "Large")]
    x = np.arange(3)
    width = 0.34
    for ax, dataset in zip(axes, ("VOC2007", "GRDDC2020")):
        baseline = next(r for r in data if r["dataset"] == dataset and r["model"] == "PD-YOLO")
        enhanced = next(r for r in data if r["dataset"] == dataset and r["model"] == "EPD-YOLO")
        b1 = ax.bar(x - width / 2, [float(baseline[k]) for k, _ in categories], width,
                    color="#4c78a8", label="PD-YOLO")
        b2 = ax.bar(x + width / 2, [float(enhanced[k]) for k, _ in categories], width,
                    color="#e45756", label="EPD-YOLO")
        ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=1)
        ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=1)
        ax.set_title(dataset)
        ax.set_xticks(x, [label for _, label in categories])
        ax.set_ylabel("AP (%) as reported")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    mark_figure(fig, "Reported results: object-scale AP")
    save(fig, "scale_ap_reported.png")


def weather_ap():
    data = rows("weather_ap50.csv")
    conditions = ["Clear", "Rain", "Fog", "Snow", "Low light"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dataset in zip(axes, ("VOC2007", "GRDDC2020")):
        subset = [r for r in data if r["dataset"] == dataset and r["condition"] in conditions]
        subset.sort(key=lambda r: conditions.index(r["condition"]))
        ax.plot(conditions, [float(r["pd_yolo_ap50_percent"]) for r in subset],
                marker="o", label="PD-YOLO", color="#4c78a8")
        ax.plot(conditions, [float(r["epd_yolo_ap50_percent"]) for r in subset],
                marker="o", label="EPD-YOLO", color="#e45756")
        ax.set_title(dataset)
        ax.set_ylabel("AP@0.5 (%) as reported")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    mark_figure(fig, "Reported results: weather-stratified AP@0.5")
    save(fig, "weather_ap50_reported.png")


def localization_ap():
    data = rows("localization_ap.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    metrics = [("ap50_percent", "AP@0.5"), ("ap75_percent", "AP@0.75"), ("ap50_95_percent", "AP@0.5:0.95")]
    x = np.arange(len(metrics))
    width = 0.34
    for ax, dataset in zip(axes, ("VOC2007", "GRDDC2020")):
        baseline = next(r for r in data if r["dataset"] == dataset and r["model"] == "PD-YOLO")
        enhanced = next(r for r in data if r["dataset"] == dataset and r["model"] == "EPD-YOLO")
        ax.bar(x - width / 2, [float(baseline[k]) for k, _ in metrics], width,
               color="#4c78a8", label="PD-YOLO")
        ax.bar(x + width / 2, [float(enhanced[k]) for k, _ in metrics], width,
               color="#e45756", label="EPD-YOLO")
        ax.set_title(dataset)
        ax.set_xticks(x, [name for _, name in metrics])
        ax.set_ylabel("AP (%) as reported")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    mark_figure(fig, "Reported results: localization-sensitive AP")
    save(fig, "localization_ap_reported.png")


def efficiency_comparison():
    data = rows("efficiency_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    datasets = ("VOC2007", "GRDDC2020")
    models = ("YOLOv8n", "PD-YOLO", "EPD-YOLO")
    colors = ("#9aa0a6", "#4c78a8", "#e45756")
    x = np.arange(len(models))
    for ax, dataset in zip(axes, datasets):
        subset = [next(r for r in data if r["dataset"] == dataset and r["model"] == model) for model in models]
        left = ax
        right = ax.twinx()
        width = 0.34
        bars = left.bar(x - width / 2, [float(r["gflops"]) for r in subset], width,
                        color=colors, alpha=0.9, label="GFLOPs")
        right.bar(x + width / 2, [float(r["fps"]) for r in subset], width,
                   color=colors, alpha=0.35, hatch="//", label="FPS")
        left.bar_label(bars, fmt="%.1f", fontsize=7, padding=1)
        left.set_title(dataset)
        left.set_xticks(x, ["YOLOv8n", "PD-YOLO", "EPD-YOLO"], rotation=15)
        left.set_ylabel("GFLOPs (reported)")
        right.set_ylabel("FPS (reported)")
        left.grid(axis="y", alpha=0.2)
        handles1, labels1 = left.get_legend_handles_labels()
        handles2, labels2 = right.get_legend_handles_labels()
        left.legend(handles1 + handles2, labels1 + labels2, frameon=False, fontsize=8)
    mark_figure(fig, "Reported results: compute and throughput")
    save(fig, "efficiency_comparison_reported.png")


if __name__ == "__main__":
    dataset_comparison()
    component_ablation()
    scale_ap()
    weather_ap()
    localization_ap()
    efficiency_comparison()
    print("Generated result charts under", ROOT)
