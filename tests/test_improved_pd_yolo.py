import numpy as np
import pytest
import torch
import yaml
from pathlib import Path

from ultralytics.data.augment import WeatherAugmentation
from ultralytics.utils.metrics import bbox_iou, compute_ap
from ultralytics.utils.loss import BboxLoss
from ultralytics.nn.extra_modules.DCSP import DCNv2
from ultralytics.utils import torch_utils


ROOT = Path(__file__).resolve().parents[1]
P2_YAML = ROOT / "ultralytics/cfg/models/YOLOv8n+C3_DCN+DCSP+attention+P2.yaml"


def test_p2_yaml_preserves_backbone_and_declares_four_scales():
    with open(P2_YAML, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert len(config["backbone"]) == 11
    assert config["head"][-1][0] == [19, 22, 25, 28]
    assert config["head"][-1][2] == "Detect"


@pytest.mark.parametrize(
    "pred,target",
    [
        ([10, 10, 30, 30], [10, 10, 30, 30]),
        ([10, 10, 20, 20], [50, 50, 80, 80]),
        ([10, 10, 40, 40], [30, 20, 60, 50]),
        ([1, 1, 1.01, 1.01], [1, 1, 1.02, 1.02]),
    ],
)
def test_siou_loss_cases_finite(pred, target):
    pred = torch.tensor([pred], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([target], dtype=torch.float32)
    siou = bbox_iou(pred, target, xywh=False, SIoU=True)
    loss = (1 - siou).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_siou_identical_box_lowest_loss():
    a = torch.tensor([[10, 10, 30, 30]], dtype=torch.float32)
    b = torch.tensor([[10, 10, 30, 30]], dtype=torch.float32)
    c = torch.tensor([[50, 50, 70, 70]], dtype=torch.float32)
    identical = 1 - bbox_iou(a, b, xywh=False, SIoU=True)
    separated = 1 - bbox_iou(a, c, xywh=False, SIoU=True)
    assert identical.item() < separated.item()


def test_compute_ap_works_with_current_numpy():
    ap, _, _ = compute_ap(np.array([0.5, 1.0]), np.array([1.0, 0.5]))
    assert np.isfinite(ap)
    assert 0.0 <= ap <= 1.0


def test_bbox_loss_configuration_selection():
    ciou = BboxLoss(reg_max=16, use_dfl=True, box_loss_type="ciou")
    siou = BboxLoss(reg_max=16, use_dfl=True, box_loss_type="siou")
    assert ciou.box_loss_type == "ciou"
    assert siou.box_loss_type == "siou"


def test_flops_profiler_skips_custom_deformable_convolution(monkeypatch):
    class ProfilerMustNotRun:
        @staticmethod
        def profile(*args, **kwargs):
            raise AssertionError("THOP must not execute for custom deformable convolution")

    monkeypatch.setattr(torch_utils, "thop", ProfilerMustNotRun)
    model = torch.nn.Sequential(DCNv2(1, 1, kernel_size=1))
    assert torch_utils.get_flops(model, imgsz=32) == 0.0


def test_weather_reproducible_and_preserves_boxes():
    image = np.full((64, 96, 3), 120, dtype=np.uint8)
    labels = {"img": image.copy(), "cls": np.array([0, 1]), "instances": object()}
    boxes_before = np.array([[0.2, 0.3, 0.4, 0.5]])
    aug1 = WeatherAugmentation(enabled=True, p=1.0, fog_prob=1.0, rain_prob=0.0,
                               shadows_prob=0.0, low_light_prob=0.0, motion_blur_prob=0.0,
                               brightness_prob=0.0)
    aug2 = WeatherAugmentation(enabled=True, p=1.0, fog_prob=1.0, rain_prob=0.0,
                               shadows_prob=0.0, low_light_prob=0.0, motion_blur_prob=0.0,
                               brightness_prob=0.0)
    out1 = aug1({"img": image.copy(), "cls": labels["cls"].copy(), "bboxes": boxes_before.copy()})
    out2 = aug2({"img": image.copy(), "cls": labels["cls"].copy(), "bboxes": boxes_before.copy()})
    assert np.array_equal(out1["img"], out2["img"])
    assert out1["img"].dtype == np.uint8
    assert out1["img"].shape == image.shape
    assert np.array_equal(out1["bboxes"], boxes_before)
    assert np.array_equal(out1["cls"], labels["cls"])
