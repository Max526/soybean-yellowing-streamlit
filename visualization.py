# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import streamlit as st

from config import AnalysisResult


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main-title { font-size: 2.2rem; font-weight: 800; margin-bottom: 0.2rem; }
        .sub-title { color: #64748b; font-size: 1.05rem; margin-bottom: 1.2rem; }
        .metric-card { border: 1px solid #e2e8f0; border-radius: 18px; padding: 18px 20px; background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06); }
        .metric-label { color: #64748b; font-size: 0.92rem; margin-bottom: 4px; }
        .metric-value { color: #0f172a; font-size: 1.7rem; font-weight: 800; }
        .diagnosis-box { border-radius: 18px; padding: 18px 20px; margin-top: 8px; border: 1px solid rgba(15, 23, 42, 0.08); }
        .healthy { background: #ecfdf5; color: #065f46; }
        .mild { background: #fefce8; color: #854d0e; }
        .moderate { background: #fff7ed; color: #9a3412; }
        .severe { background: #fef2f2; color: #991b1b; }
        .small-note { color: #64748b; font-size: 0.9rem; }
        .warn-note { color: #92400e; background: #fffbeb; border: 1px solid #fde68a; border-radius: 12px; padding: 10px 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_st_image(image: Any, caption: str) -> None:
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image, caption=caption, use_column_width=True)


def render_metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def draw_detection_summary(image_bgr: np.ndarray, results: list[AnalysisResult]) -> np.ndarray:
    canvas = image_bgr.copy()
    for item in results:
        x1, y1, x2, y2 = item.bbox
        color = (0, 180, 255) if item.yellow_ratio >= 25 else (40, 180, 60)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        label = f"Leaf {item.leaf_id}: {item.yellow_ratio:.1f}%"
        label_width = min(260, max(190, len(label) * 13))
        cv2.rectangle(canvas, (x1, max(y1 - 30, 0)), (min(x1 + label_width, canvas.shape[1]), y1), color, -1)
        cv2.putText(canvas, label, (x1 + 8, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return canvas


def render_model_info(model_info: dict[str, str]) -> None:
    with st.expander("模型資訊與 nano4 訓練結果", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric("模型", model_info["model"])
        with c2:
            render_metric("訓練平台", model_info["training_platform"])
        with c3:
            render_metric("資料量", model_info["dataset_size"])
        with c4:
            render_metric("mAP50", model_info["map50"])
        st.caption(f"Precision {model_info['precision']}｜Recall {model_info['recall']}｜mAP50-95 {model_info['map50_95']}")
