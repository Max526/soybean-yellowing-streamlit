# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd

from config import AnalysisResult, HSVConfig, RuntimeConfig
from image_processing import preprocess_image
from model_utils import get_leaf_detections


def clean_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 0:
        return mask
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def analyze_leaf_color(leaf_bgr: np.ndarray, config: HSVConfig) -> tuple[float, float, float, int, np.ndarray]:
    hsv = cv2.cvtColor(leaf_bgr, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, np.array(config.yellow_lower), np.array(config.yellow_upper))
    green_mask = cv2.inRange(hsv, np.array(config.green_lower), np.array(config.green_upper))
    valid_mask = ((hsv[:, :, 1] >= config.min_leaf_saturation) & (hsv[:, :, 2] >= config.min_leaf_value)).astype(np.uint8) * 255

    yellow_mask = clean_mask(cv2.bitwise_and(yellow_mask, valid_mask), config.morph_kernel_size)
    green_mask = clean_mask(cv2.bitwise_and(green_mask, valid_mask), config.morph_kernel_size)

    leaf_color_mask = cv2.bitwise_or(yellow_mask, green_mask)
    leaf_area = int(np.count_nonzero(leaf_color_mask))
    yellow_area = int(np.count_nonzero(cv2.bitwise_and(yellow_mask, leaf_color_mask)))
    green_area = int(np.count_nonzero(green_mask))

    if leaf_area == 0:
        return 0.0, 0.0, 0.0, 0, yellow_mask
    return yellow_area / leaf_area * 100, green_area / leaf_area * 100, yellow_area / max(green_area, 1), leaf_area, yellow_mask


def diagnose(yellow_ratio: float) -> tuple[str, str, str]:
    if yellow_ratio < 10:
        return "健康 / 正常", "healthy", "葉片黃化比例低，整體狀態接近正常。"
    if yellow_ratio < 25:
        return "輕度黃化", "mild", "葉片已有輕微黃化，建議觀察是否持續擴大，並檢查水分與養分狀態。"
    if yellow_ratio < 45:
        return "中度黃化", "moderate", "葉片黃化明顯，可能與淹水逆境、缺氮、根系缺氧或病害壓力有關。"
    return "嚴重黃化", "severe", "葉片黃化比例偏高，建議立即檢查田間排水、根系狀態與病蟲害情形。"


def make_mask_overlay(leaf_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    yellow_layer = np.zeros_like(leaf_bgr)
    yellow_layer[:, :] = (0, 255, 255)
    blended = cv2.addWeighted(leaf_bgr, 0.45, yellow_layer, 0.55, 0)
    return np.where(mask[:, :, None] > 0, blended, leaf_bgr)


def analyze_image(filename: str, image_bgr: np.ndarray, model: Any | None, config: RuntimeConfig) -> tuple[list[AnalysisResult], np.ndarray, list[np.ndarray]]:
    processed_bgr = preprocess_image(image_bgr, config.apply_gray_world, config.apply_clahe)
    if model is None:
        h, w = processed_bgr.shape[:2]
        detections = [{"bbox": (0, 0, w, h), "confidence": 1.0, "class_id": 0, "area": w * h}]
    else:
        detections = get_leaf_detections(model, processed_bgr, config.conf, config.iou, config.max_det)
    if config.only_largest_leaf and detections:
        detections = detections[:1]

    results: list[AnalysisResult] = []
    overlays: list[np.ndarray] = []
    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["bbox"]
        leaf_crop = processed_bgr[y1:y2, x1:x2]
        if leaf_crop.size == 0:
            continue
        yellow_ratio, green_ratio, yg_ratio, valid_area, yellow_mask = analyze_leaf_color(leaf_crop, config.hsv)
        diagnosis, _, suggestion = diagnose(yellow_ratio)
        overlays.append(make_mask_overlay(leaf_crop, yellow_mask))
        results.append(AnalysisResult(filename, idx, float(det["confidence"]), float(yellow_ratio), float(green_ratio), float(yg_ratio), int(valid_area), diagnosis, suggestion, det["bbox"]))
    return results, processed_bgr, overlays


def results_to_dataframe(results: list[AnalysisResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "filename": item.filename,
            "leaf_id": item.leaf_id,
            "confidence": round(item.confidence, 4),
            "yellow_ratio_%": round(item.yellow_ratio, 2),
            "green_ratio_%": round(item.green_ratio, 2),
            "yellow_green_ratio": round(item.yellow_green_ratio, 4),
            "valid_area_px": item.valid_area,
            "diagnosis": item.diagnosis,
            "suggestion": item.suggestion,
            "bbox": item.bbox,
        }
        for item in results
    ])


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
