# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config import APP_TITLE, DEFAULT_WEIGHT_PATH, MAX_IMAGE_SIDE, MODEL_INFO, SUPPORTED_IMAGE_TYPES, HSVConfig, RuntimeConfig
from image_processing import bgr_to_rgb, load_uploaded_image, pil_to_bgr
from model_utils import load_yolo_model
from visualization import draw_detection_summary, inject_css, render_metric, render_model_info, safe_st_image
from yellowing_analysis import analyze_image, dataframe_to_csv_bytes, diagnose, results_to_dataframe

st.set_page_config(page_title=APP_TITLE, page_icon="🌱", layout="wide", initial_sidebar_state="expanded")


def build_hsv_config() -> HSVConfig:
    st.sidebar.markdown("### 黃化判斷參數 HSV")
    st.sidebar.caption("HSV 參數會影響哪些像素被視為黃色或綠色。若照片偏黃、偏暗或光線不均，可小幅調整。")

    with st.sidebar.expander("黃色像素判斷範圍", expanded=False):
        y_h_min = st.slider("黃色色相下限", 0, 179, 18)
        y_h_max = st.slider("黃色色相上限", 0, 179, 38)
        y_s_min = st.slider("黃色最低飽和度", 0, 255, 60)
        y_v_min = st.slider("黃色最低亮度", 0, 255, 80)

    with st.sidebar.expander("綠色像素判斷範圍", expanded=False):
        g_h_min = st.slider("綠色色相下限", 0, 179, 35)
        g_h_max = st.slider("綠色色相上限", 0, 179, 90)
        g_s_min = st.slider("綠色最低飽和度", 0, 255, 30)
        g_v_min = st.slider("綠色最低亮度", 0, 255, 45)

    with st.sidebar.expander("有效葉片像素過濾", expanded=False):
        min_sat = st.slider("有效像素最低飽和度", 0, 255, 40)
        min_val = st.slider("有效像素最低亮度", 0, 255, 45)
        morph_kernel_size = st.slider("遮罩去雜訊強度", 0, 9, 3, 2)

    return HSVConfig(
        yellow_lower=(y_h_min, y_s_min, y_v_min),
        yellow_upper=(y_h_max, 255, 255),
        green_lower=(g_h_min, g_s_min, g_v_min),
        green_upper=(g_h_max, 255, 255),
        min_leaf_saturation=min_sat,
        min_leaf_value=min_val,
        morph_kernel_size=morph_kernel_size,
    )


def render_sidebar() -> RuntimeConfig:
    st.sidebar.title("⚙️ 分析參數設定")
    st.sidebar.caption("這些設定會影響葉片偵測與黃化比例計算；若不確定，建議使用預設值。")

    weight_path = st.sidebar.text_input("YOLO 權重路徑", DEFAULT_WEIGHT_PATH)
    st.sidebar.markdown("### YOLO 葉片偵測設定")
    conf = st.sidebar.slider("葉片偵測信心值", 0.05, 0.95, 0.25, 0.05)
    iou = st.sidebar.slider("重疊框合併門檻 IoU", 0.10, 0.90, 0.45, 0.05)
    max_det = st.sidebar.slider("最多偵測葉片數", 1, 50, 10, 1)
    only_largest_leaf = st.sidebar.checkbox("只分析最大葉片", value=False)

    st.sidebar.markdown("### 影像前處理")
    resize_long_side = st.sidebar.slider("最大影像邊長", 640, 2560, MAX_IMAGE_SIDE, 160)
    apply_gray_world = st.sidebar.checkbox("自動白平衡", value=True)
    apply_clahe = st.sidebar.checkbox("亮度對比增強", value=False)
    hsv_config = build_hsv_config()

    st.sidebar.markdown("---")
    st.sidebar.info("若模型權重不存在，平台會暫時以整張圖片進行 HSV 分析，僅供測試介面。")
    return RuntimeConfig(weight_path, conf, iou, max_det, only_largest_leaf, resize_long_side, apply_gray_world, apply_clahe, hsv_config)


def render_header() -> None:
    st.markdown(f'<div class="main-title">🌱 {APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">面向農業研究、教學展示與田間初步篩檢的黃化程度量化工具</div>', unsafe_allow_html=True)
    render_model_info(MODEL_INFO)


def render_empty_state() -> None:
    st.info("請上傳一張或多張葉片圖片開始分析。")
    st.markdown(
        """
        #### 平台流程
        1. 使用 YOLO 找出葉片位置  
        2. 對葉片區域做白平衡 / 亮度前處理  
        3. 在 HSV 色彩空間計算黃色比例、綠色比例與黃綠比  
        4. 依黃化比例輸出健康、輕度、中度或嚴重黃化分級  
        5. 匯出 CSV，方便後續統計或放進研究報告
        """
    )


def image_to_bgr_or_none(uploaded_file: Any, config: RuntimeConfig):
    try:
        image = load_uploaded_image(uploaded_file)
        return image, pil_to_bgr(image, config.resize_long_side)
    except ValueError as exc:
        st.error(str(exc))
        return None, None


def render_single_image_result(uploaded_file: Any, model: Any | None, config: RuntimeConfig) -> list:
    image, image_bgr = image_to_bgr_or_none(uploaded_file, config)
    if image is None or image_bgr is None:
        return []

    start = time.time()
    results, processed_bgr, overlays = analyze_image(uploaded_file.name, image_bgr, model, config)
    elapsed = time.time() - start

    if not results:
        st.warning("沒有偵測到葉片。可以降低 confidence，或確認圖片中葉片是否清楚。")
        safe_st_image(image, "原始圖片")
        return []

    df = results_to_dataframe(results)
    avg_yellow = float(df["yellow_ratio_%"].mean())
    avg_green = float(df["green_ratio_%"].mean())
    main_diagnosis, diag_class, suggestion = diagnose(avg_yellow)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric("偵測葉片數", f"{len(results)}")
    with c2:
        render_metric("平均黃化比例", f"{avg_yellow:.1f}%")
    with c3:
        render_metric("平均綠色比例", f"{avg_green:.1f}%")
    with c4:
        render_metric("分析時間", f"{elapsed:.2f}s")

    st.markdown(f'<div class="diagnosis-box {diag_class}"><h3 style="margin:0 0 6px 0;">黃化分級：{main_diagnosis}</h3><div>{suggestion}</div></div>', unsafe_allow_html=True)
    if model is None:
        st.markdown('<div class="warn-note">目前未載入 YOLO 權重，因此結果是「整張圖片 HSV 測試分析」。</div>', unsafe_allow_html=True)

    annotated = draw_detection_summary(processed_bgr, results)
    st.markdown("### 影像分析結果")
    col1, col2, col3 = st.columns(3)
    with col1:
        safe_st_image(image, "原始圖片")
    with col2:
        safe_st_image(bgr_to_rgb(annotated), "YOLO 偵測與黃化比例")
    with col3:
        if overlays:
            safe_st_image(bgr_to_rgb(overlays[0]), "葉片 ROI 黃化遮罩")
        else:
            st.info("沒有可顯示的遮罩。")

    with st.expander("查看每片葉片詳細數據", expanded=True):
        st.dataframe(df, use_container_width=True)
        st.download_button("下載本張圖片分析 CSV", data=dataframe_to_csv_bytes(df), file_name=f"{Path(uploaded_file.name).stem}_analysis.csv", mime="text/csv")
    return results


def render_batch_results(uploaded_files: list[Any], model: Any | None, config: RuntimeConfig) -> list:
    st.markdown("### 批次分析結果")
    progress = st.progress(0)
    status = st.empty()
    all_results = []
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        status.write(f"正在分析：{uploaded_file.name} ({idx}/{len(uploaded_files)})")
        _, image_bgr = image_to_bgr_or_none(uploaded_file, config)
        if image_bgr is not None:
            results, _, _ = analyze_image(uploaded_file.name, image_bgr, model, config)
            all_results.extend(results)
        progress.progress(idx / len(uploaded_files))
    status.success("批次分析完成")

    if not all_results:
        st.warning("批次圖片中沒有偵測到葉片。")
        return []

    df = results_to_dataframe(all_results)
    summary = df.groupby("filename", as_index=False).agg(leaf_count=("leaf_id", "count"), avg_yellow_ratio=("yellow_ratio_%", "mean"), avg_green_ratio=("green_ratio_%", "mean"))
    summary["avg_yellow_ratio"] = summary["avg_yellow_ratio"].round(2)
    summary["avg_green_ratio"] = summary["avg_green_ratio"].round(2)
    summary["diagnosis"] = summary["avg_yellow_ratio"].apply(lambda x: diagnose(float(x))[0])

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric("圖片數", str(len(uploaded_files)))
    with c2:
        render_metric("總葉片數", str(len(all_results)))
    with c3:
        render_metric("整體平均黃化", f"{summary['avg_yellow_ratio'].mean():.1f}%")

    st.markdown("#### 每張圖片摘要")
    st.dataframe(summary, use_container_width=True)
    st.markdown("#### 每片葉片詳細資料")
    st.dataframe(df, use_container_width=True)
    st.download_button("下載完整批次分析 CSV", data=dataframe_to_csv_bytes(df), file_name="batch_leaf_yellowing_analysis.csv", mime="text/csv")
    return all_results


def main() -> None:
    inject_css()
    render_header()
    config = render_sidebar()
    model = load_yolo_model(config.weight_path)

    st.markdown("### 上傳葉片圖片")
    st.caption("本平台提供黃化比例量化，不直接判定病因。")
    uploaded_files = st.file_uploader("支援 JPG / JPEG / PNG / BMP，可一次上傳多張。", type=SUPPORTED_IMAGE_TYPES, accept_multiple_files=True)

    if not uploaded_files:
        render_empty_state()
    elif len(uploaded_files) == 1:
        render_single_image_result(uploaded_files[0], model, config)
    else:
        render_batch_results(list(uploaded_files), model, config)

    st.markdown("---")
    st.markdown('<div class="small-note">提醒：本系統提供黃化比例量化與影像輔助判斷，不等同完整病害診斷。</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
