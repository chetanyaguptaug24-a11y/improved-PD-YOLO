# Improved PD-YOLO

## Files

- `ultralytics/cfg/models/YOLOv8n+C3_DCN+DCSP+attention+P2.yaml`
- `ultralytics/utils/loss.py`
- `ultralytics/utils/metrics.py`
- `ultralytics/data/augment.py`
- `ultralytics/cfg/default.yaml`
- `ultralytics/cfg/__init__.py`
- `train_improved_pd_yolo.py`
- `evaluate_improved_pd_yolo.py`
- `tests/test_improved_pd_yolo.py`

## P2 architecture

The P2 YAML preserves the original C3D, DCSP, and BiLevelRoutingAttention backbone. Its head adds stride-4 P2 fusion and
detects at four feature levels. Detection feature indices are `[19, 22, 25, 28]`, corresponding to strides
`[4, 8, 16, 32]`.

## SIoU

The existing `bbox_iou(..., SIoU=True)` implementation is used as the regression objective for foreground assignments.
Classification and DFL remain unchanged. Set `box_loss_type: siou` to enable it; the default remains `ciou`.

## Weather augmentation

`WeatherAugmentation` is inserted at the end of the training-only `v8_transforms` pipeline. It changes only `labels["img"]`
and does not modify instances or normalized boxes. Validation and test datasets call the same dataset code with
`augment=False`, so they do not receive weather effects.

Effect selection probabilities are independent: first, `weather_prob` decides whether a sample receives an effect; then
each enabled effect probability decides whether it enters a candidate list. Exactly one candidate is sampled for each
eligible image.

## Train

```powershell
python train_improved_pd_yolo.py --epochs 100 --imgsz 640 --batch 8 --optimizer SGD --seed 42 --weather-aug --weather-prob 0.5
```

Baseline:

```powershell
python train.py
```

Resume:

```powershell
python train_improved_pd_yolo.py --resume runs/train/improved_pd_yolo/weights/last.pt
```

## Evaluate

```powershell
python evaluate_improved_pd_yolo.py --data ultralytics/cfg/datasets/Bump.yaml --split test --device cpu Original=ultralytics/cfg/models/YOLOv8n+C3_DCN+DCSP+attention.yaml Improved=ultralytics/cfg/models/YOLOv8n+C3_DCN+DCSP+attention+P2.yaml
```

Run `pytest tests/test_improved_pd_yolo.py -q` after Windows application-control permits loading the PyTorch runtime.

## Supplied external-machine results

The supplied notebook includes a measured, one-epoch CPU pilot for baseline PD-YOLO versus PD-YOLO + P2 on 67 VOC2007
test images. The transcribed values, source hashes, protocol, environment, and limitations are in [`results/README.md`](results/README.md).
This pilot has very low AP and does not support a claim of useful accuracy or model superiority.

Additional results confirmed by the user are recorded separately in [`results/reported/`](results/reported/) with
CSV tables and comparison charts. The user reports that these results were independently reproduced on a separate
GPU-equipped PC; reproduction logs and checkpoints are not included in this repository.
