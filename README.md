# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


@st.cache_resource(show_spinner=False)
def load_yolo_model(weight_path: str) -> Any | None:
    if YOLO is None:
        st.error("尚未安裝 ultralytics。請確認 requirements.txt 已包含 ultralytics。")
        return None
    path = Path(weight_path)
    if not path.exists():
        st.warning(f"找不到模型權重：{weight_path}。目前會以整張圖片做 HSV 測試分析，正式使用請放入 YOLO 權重。")
        return None
    try:
        return YOLO(str(path))
    except Exception as exc:
        st.error(f"模型載入失敗：{exc}")
        return None


def get_leaf_detections(model: Any, image_bgr: np.ndarray, conf: float, iou: float, max_det: int) -> list[dict[str, Any]]:
    results = model.predict(image_bgr, conf=conf, iou=iou, max_det=max_det, verbose=False)
    if not results or results[0].boxes is None:
        return []
    detections: list[dict[str, Any]] = []
    height, width = image_bgr.shape[:2]
    for box in results[0].boxes:
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(int)
        x1, y1, x2, y2 = clamp_bbox(*xyxy, width=width, height=height)
        if x2 <= x1 or y2 <= y1:
            continue
        confidence = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
        cls_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else -1
        area = (x2 - x1) * (y2 - y1)
        detections.append({"bbox": (x1, y1, x2, y2), "confidence": confidence, "class_id": cls_id, "area": area})
    detections.sort(key=lambda item: item["area"], reverse=True)
    return detections
