# Improved PD-YOLO

An Ultralytics-based pothole detector that extends the PD-YOLO design with a
stride-4 P2 detection head, alongside the existing P3/P4/P5 heads. The repository
also includes custom convolution/attention components, training utilities, and
small-object and adverse-weather augmentation experiments.

## What is included

- Original PD-YOLO and P2 model configurations in
  [`ultralytics/cfg/models/`](ultralytics/cfg/models/).
- Custom model components and training code in [`ultralytics/`](ultralytics/).
- A side-by-side demonstration notebook:
  [`Improved_PD_YOLO_Experiments.ipynb`](Improved_PD_YOLO_Experiments.ipynb).
- Matched local CPU comparison artifacts in [`runs/small_demo/`](runs/small_demo/):
  a scorecard, rendered predictions, and the final baseline and P2 checkpoints.
- Additional independently reproduced GPU results and charts in
  [`results/reported/`](results/reported/). These are separate from the local
  CPU demonstration and should not be combined into one direct comparison.

## Environment setup

The project is tested with Python 3.11. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name improved-pd-yolo --display-name "Python (.venv) - Improved PD-YOLO"
```

The comparison notebook is configured for CPU inference. Training and validation
are substantially faster with a supported NVIDIA GPU.

## View the comparison

Open the notebook and select the `Python (.venv) - Improved PD-YOLO` kernel.
Run its final **Matched baseline vs. P2 comparison** section to reproduce the
metrics and prediction panels. The saved presentation artifacts are also
available directly:

- [`comparison_examples.png`](runs/small_demo/comparison_examples.png)
- [`comparison_summary.csv`](runs/small_demo/comparison_summary.csv)
- Original PD-YOLO checkpoint:
  [`best.pt`](runs/small_demo/original_pd_64img_refine30_cpu/weights/best.pt)
- Improved P2 checkpoint:
  [`best.pt`](runs/small_demo/improved_p2_64img_refine30_cpu/weights/best.pt)

The demo is a one-class pothole experiment using a 64-image training subset, a
16-image validation subset, and the full 67-image held-out test split (171
ground-truth boxes). Both models used the same local `yolov8n.pt` initialization,
320-pixel input, CPU, seed, and 20-epoch plus 30-epoch refinement schedule.
Precision and recall below use a shared confidence threshold of 0.25 and
one-to-one matching at IoU 0.50; mAP is confidence-swept.

| Model | Parameters (M) | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | TP / FP / FN |
|---|---:|---:|---:|---:|---:|---:|
| Original PD-YOLO | 3.679 | 0.608 | 0.181 | 0.2120 | 0.0644 | 31 / 20 / 140 |
| Improved PD-YOLO (P2) | 3.630 | 0.444 | 0.117 | 0.1353 | 0.0408 | 20 / 25 / 151 |

This small CPU run demonstrates the P2 model and its predictions, but does not
show an accuracy gain over the original in this experiment. The P2 configuration
has approximately 1.3% fewer parameters, while adding a higher-resolution
detection head. The result should be treated as a limited small-subset
comparison, not a general claim about the architecture; broader conclusions
require matched full training, multiple seeds, and independent evaluation.

## Results provenance

The tables and charts under [`results/reported/`](results/reported/) contain
additional results that were independently reproduced on a separate GPU-equipped
machine. They use a different experiment protocol from the small CPU demo above.
The locally saved checkpoints and CSVs in `runs/small_demo/` belong only to the
CPU demonstration.

## Tests

Run the focused custom-model tests with:

```powershell
python -m pytest -q tests/test_improved_pd_yolo.py
```
