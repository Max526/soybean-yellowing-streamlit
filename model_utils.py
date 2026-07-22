# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

APP_TITLE = "大豆葉片黃化程度分析平台"
DEFAULT_WEIGHT_PATH = "weights/best.pt"
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp"]
MAX_IMAGE_SIDE = 1920

MODEL_INFO = {
    "model": "YOLO11s Detect",
    "training_platform": "NCHC nano4",
    "dataset_size": "250 images",
    "precision": "0.845",
    "recall": "0.960",
    "map50": "0.954",
    "map50_95": "0.574",
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
