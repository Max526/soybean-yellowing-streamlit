# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

APP_TITLE = "大豆葉片黃化程度分析平台｜春秋季正式版"
DEFAULT_WEIGHT_PATH = "weights/best_spring_autumn.pt"
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp"]
MAX_IMAGE_SIDE = 1920

MODEL_INFO = {
    "model": "YOLO11n Detect（春秋季正式版）",
    "training_platform": "NCHC nano4 / H200 GPU",
    "dataset_size": "557 images（春季 307 + 秋季 250）",
    "split": "train 445 / val 55 / test 57",
    "precision": "0.999",
    "recall": "1.000",
    "map50": "0.995",
    "map50_95": "0.967",
    "weights": "weights/best_spring_autumn.pt",
}


@dataclass(frozen=True)
class HSVConfig:
    yellow_lower: tuple[int, int, int]
    yellow_upper: tuple[int, int, int]
    green_lower: tuple[int, int, int]
    green_upper: tuple[int, int, int]
    min_leaf_saturation: int
    min_leaf_value: int
    morph_kernel_size: int


@dataclass(frozen=True)
class RuntimeConfig:
    weight_path: str
    conf: float
    iou: float
    max_det: int
    only_largest_leaf: bool
    resize_long_side: int
    apply_gray_world: bool
    apply_clahe: bool
    hsv: HSVConfig


@dataclass(frozen=True)
class AnalysisResult:
    filename: str
    leaf_id: int
    confidence: float
    yellow_ratio: float
    green_ratio: float
    yellow_green_ratio: float
    valid_area: int
    diagnosis: str
    suggestion: str
    bbox: tuple[int, int, int, int]
