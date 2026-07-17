from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


# =========================
# 基本設定
# =========================
APP_TITLE = "大豆葉片黃化智慧診斷平台"
DEFAULT_WEIGHT_PATH = "weights/best.pt"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# 資料結構
# =========================
@dataclass
class HSVConfig:
    yellow_lower: tuple[int, int, int]
    yellow_upper: tuple[int, int, int]
    green_lower: tuple[int, int, int]
    green_upper: tuple[int, int, int]
    min_leaf_saturation: int
    min_leaf_value: int


@dataclass
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


# =========================
# 樣式
# =========================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            color: #64748b;
            font-size: 1.05rem;
            margin-bottom: 1.2rem;
        }
        .metric-card {
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 18px 20px;
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }
        .metric-label {
            color: #64748b;
            font-size: 0.92rem;
            margin-bottom: 4px;
        }
        .metric-value {
            color: #0f172a;
            font-size: 1.7rem;
            font-weight: 800;
        }
        .diagnosis-box {
            border-radius: 18px;
            padding: 18px 20px;
            margin-top: 8px;
            border: 1px solid rgba(15, 23, 42, 0.08);
        }
        .healthy { background: #ecfdf5; color: #065f46; }
        .mild { background: #fefce8; color: #854d0e; }
        .moderate { background: #fff7ed; color: #9a3412; }
        .severe { background: #fef2f2; color: #991b1b; }
        .small-note {
            color: #64748b;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# 模型與影像工具
# =========================
@st.cache_resource(show_spinner=False)
def load_yolo_model(weight_path: str) -> Any | None:
    if YOLO is None:
        st.error("尚未安裝 ultralytics。請先執行：pip install ultralytics")
        return None

    path = Path(weight_path)
    if not path.exists():
        st.warning(f"找不到模型權重：{weight_path}。請把 best.pt 放到 weights/best.pt，或在側邊欄改路徑。")
        return None

    try:
        return YOLO(str(path))
    except Exception as exc:
        st.error(f"模型載入失敗：{exc}")
        return None


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def get_leaf_detections(model: Any, image_bgr: np.ndarray, conf: float, iou: float, max_det: int) -> list[dict[str, Any]]:
    results = model.predict(image_bgr, conf=conf, iou=iou, max_det=max_det, verbose=False)
    if not results:
        return []

    result = results[0]
    detections: list[dict[str, Any]] = []
    height, width = image_bgr.shape[:2]

    if result.boxes is None:
        return detections

    for box in result.boxes:
        xyxy = box.xyxy[0].detach().cpu().numpy().astype(int)
        x1, y1, x2, y2 = clamp_bbox(*xyxy, width=width, height=height)
        if x2 <= x1 or y2 <= y1:
            continue
        confidence = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
        cls_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else -1
        area = (x2 - x1) * (y2 - y1)
        detections.append(
            {
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "class_id": cls_id,
                "area": area,
            }
        )

    detections.sort(key=lambda item: item["area"], reverse=True)
    return detections


# =========================
# 黃化分析
# =========================
def build_hsv_config() -> HSVConfig:
    st.sidebar.markdown("### HSV 黃化參數")

    with st.sidebar.expander("黃色範圍", expanded=False):
        y_h_min = st.slider("Yellow Hue min", 0, 179, 15)
        y_h_max = st.slider("Yellow Hue max", 0, 179, 40)
        y_s_min = st.slider("Yellow Saturation min", 0, 255, 35)
        y_v_min = st.slider("Yellow Value min", 0, 255, 70)

    with st.sidebar.expander("綠色範圍", expanded=False):
        g_h_min = st.slider("Green Hue min", 0, 179, 35)
        g_h_max = st.slider("Green Hue max", 0, 179, 90)
        g_s_min = st.slider("Green Saturation min", 0, 255, 30)
        g_v_min = st.slider("Green Value min", 0, 255, 45)

    with st.sidebar.expander("有效葉片像素過濾", expanded=False):
        min_sat = st.slider("最低飽和度", 0, 255, 25)
        min_val = st.slider("最低亮度", 0, 255, 35)

    return HSVConfig(
        yellow_lower=(y_h_min, y_s_min, y_v_min),
        yellow_upper=(y_h_max, 255, 255),
        green_lower=(g_h_min, g_s_min, g_v_min),
        green_upper=(g_h_max, 255, 255),
        min_leaf_saturation=min_sat,
        min_leaf_value=min_val,
    )


def analyze_leaf_color(leaf_bgr: np.ndarray, config: HSVConfig) -> tuple[float, float, float, int, np.ndarray]:
    hsv = cv2.cvtColor(leaf_bgr, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, np.array(config.yellow_lower), np.array(config.yellow_upper))
    green_mask = cv2.inRange(hsv, np.array(config.green_lower), np.array(config.green_upper))

    # 有效葉片區域：排除過暗、灰白、低飽和背景
    valid_mask = ((hsv[:, :, 1] >= config.min_leaf_saturation) & (hsv[:, :, 2] >= config.min_leaf_value)).astype(np.uint8) * 255

    yellow_mask = cv2.bitwise_and(yellow_mask, valid_mask)
    green_mask = cv2.bitwise_and(green_mask, valid_mask)

    valid_area = int(np.count_nonzero(valid_mask))
    yellow_area = int(np.count_nonzero(yellow_mask))
    green_area = int(np.count_nonzero(green_mask))

    if valid_area == 0:
        return 0.0, 0.0, 0.0, 0, yellow_mask

    yellow_ratio = yellow_area / valid_area * 100
    green_ratio = green_area / valid_area * 100
    yellow_green_ratio = yellow_area / max(green_area, 1)

    return yellow_ratio, green_ratio, yellow_green_ratio, valid_area, yellow_mask


def diagnose(yellow_ratio: float) -> tuple[str, str, str]:
    if yellow_ratio < 10:
        return "健康 / 正常", "healthy", "葉片黃化比例低，整體狀態接近正常。"
    if yellow_ratio < 25:
        return "輕度黃化", "mild", "葉片已有輕微黃化，建議觀察是否持續擴大，並檢查水分與養分狀態。"
    if yellow_ratio < 45:
        return "中度黃化", "moderate", "葉片黃化明顯，可能與淹水逆境、缺氮、根系缺氧或病害壓力有關。"
    return "嚴重黃化", "severe", "葉片黃化比例偏高，建議立即檢查田間排水、根系狀態與病蟲害情形。"


def make_mask_overlay(leaf_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = leaf_bgr.copy()
    yellow_layer = np.zeros_like(leaf_bgr)
    yellow_layer[:, :] = (0, 255, 255)
    overlay = np.where(mask[:, :, None] > 0, cv2.addWeighted(leaf_bgr, 0.45, yellow_layer, 0.55, 0), overlay)
    return overlay


def draw_detection_summary(image_bgr: np.ndarray, results: list[AnalysisResult]) -> np.ndarray:
    canvas = image_bgr.copy()
    for item in results:
        x1, y1, x2, y2 = item.bbox
        color = (0, 180, 255) if item.yellow_ratio >= 25 else (40, 180, 60)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        label = f"Leaf {item.leaf_id}: {item.yellow_ratio:.1f}%"
        cv2.rectangle(canvas, (x1, max(y1 - 30, 0)), (x1 + 220, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 8, max(y1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return canvas


# =========================
# 分析流程
# =========================
def analyze_image(
    filename: str,
    image_bgr: np.ndarray,
    model: Any | None,
    config: HSVConfig,
    conf: float,
    iou: float,
    max_det: int,
    only_largest_leaf: bool,
) -> tuple[list[AnalysisResult], np.ndarray, list[np.ndarray]]:
    if model is None:
        # 沒有 YOLO 時，退回整張圖分析，方便先測平台；此結果不能視為葉片偵測結果。
        h, w = image_bgr.shape[:2]
        detections = [{"bbox": (0, 0, w, h), "confidence": 1.0, "class_id": 0, "area": w * h}]
    else:
        detections = get_leaf_detections(model, image_bgr, conf=conf, iou=iou, max_det=max_det)

    if only_largest_leaf and detections:
        detections = detections[:1]

    results: list[AnalysisResult] = []
    overlays: list[np.ndarray] = []

    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["bbox"]
        leaf_crop = image_bgr[y1:y2, x1:x2]
        yellow_ratio, green_ratio, yg_ratio, valid_area, yellow_mask = analyze_leaf_color(leaf_crop, config)
        diagnosis, _, suggestion = diagnose(yellow_ratio)
        overlays.append(make_mask_overlay(leaf_crop, yellow_mask))
        results.append(
            AnalysisResult(
                filename=filename,
                leaf_id=idx,
                confidence=float(det["confidence"]),
                yellow_ratio=float(yellow_ratio),
                green_ratio=float(green_ratio),
                yellow_green_ratio=float(yg_ratio),
                valid_area=int(valid_area),
                diagnosis=diagnosis,
                suggestion=suggestion,
                bbox=det["bbox"],
            )
        )

    annotated = draw_detection_summary(image_bgr, results)
    return results, annotated, overlays


def results_to_dataframe(results: list[AnalysisResult]) -> pd.DataFrame:
    rows = []
    for item in results:
        rows.append(
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
        )
    return pd.DataFrame(rows)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# =========================
# UI
# =========================
def render_header() -> None:
    st.markdown(f'<div class="main-title">🌱 {APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">YOLO 葉片偵測 × HSV 黃化比例分析 × 批次 CSV 匯出</div>',
        unsafe_allow_html=True,
    )


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


def render_sidebar() -> tuple[str, float, float, int, bool, HSVConfig]:
    st.sidebar.title("⚙️ 分析設定")

    weight_path = st.sidebar.text_input("YOLO 權重路徑", DEFAULT_WEIGHT_PATH)
    st.sidebar.caption("建議將訓練好的模型放在 weights/best.pt")

    st.sidebar.markdown("### YOLO 偵測參數")
    conf = st.sidebar.slider("信心閾值 Confidence", 0.05, 0.95, 0.25, 0.05)
    iou = st.sidebar.slider("NMS IoU 閾值", 0.10, 0.90, 0.45, 0.05)
    max_det = st.sidebar.slider("最多偵測葉片數", 1, 50, 10, 1)
    only_largest_leaf = st.sidebar.checkbox("只分析最大葉片", value=False)

    config = build_hsv_config()

    st.sidebar.markdown("---")
    st.sidebar.info("若模型權重不存在，平台會暫時以整張圖片進行 HSV 分析，方便先測試介面。")

    return weight_path, conf, iou, max_det, only_largest_leaf, config


def render_single_image_result(
    uploaded_file: Any,
    model: Any | None,
    config: HSVConfig,
    conf: float,
    iou: float,
    max_det: int,
    only_largest_leaf: bool,
) -> list[AnalysisResult]:
    image = Image.open(uploaded_file)
    image_bgr = pil_to_bgr(image)

    start = time.time()
    results, annotated, overlays = analyze_image(
        filename=uploaded_file.name,
        image_bgr=image_bgr,
        model=model,
        config=config,
        conf=conf,
        iou=iou,
        max_det=max_det,
        only_largest_leaf=only_largest_leaf,
    )
    elapsed = time.time() - start

    if not results:
        st.warning("沒有偵測到葉片。可以降低 confidence，或確認圖片中葉片是否清楚。")
        st.image(image, caption="原始圖片", use_column_width=True)
        return []

    df = results_to_dataframe(results)
    avg_yellow = df["yellow_ratio_%"].mean()
    avg_green = df["green_ratio_%"].mean()
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

    st.markdown(
        f"""
        <div class="diagnosis-box {diag_class}">
            <h3 style="margin: 0 0 6px 0;">診斷結果：{main_diagnosis}</h3>
            <div>{suggestion}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 影像分析結果")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="原始圖片", use_column_width=True)
    with col2:
        st.image(bgr_to_rgb(annotated), caption="YOLO 偵測與黃化比例", use_column_width=True)
    with col3:
        if overlays:
            st.image(bgr_to_rgb(overlays[0]), caption="第 1 片葉片黃化遮罩", use_column_width=True)
        else:
            st.info("沒有可顯示的遮罩。")

    with st.expander("查看每片葉片詳細數據", expanded=True):
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "下載本張圖片分析 CSV",
            data=dataframe_to_csv_bytes(df),
            file_name=f"{Path(uploaded_file.name).stem}_analysis.csv",
            mime="text/csv",
        )

    return results


def main() -> None:
    inject_css()
    render_header()

    weight_path, conf, iou, max_det, only_largest_leaf, config = render_sidebar()
    model = load_yolo_model(weight_path)

    st.markdown("### 上傳葉片圖片")
    uploaded_files = st.file_uploader(
        "支援 JPG / JPEG / PNG，可一次上傳多張進行批次分析。",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("請上傳一張或多張葉片圖片開始分析。")
        st.markdown(
            """
            #### 平台流程
            1. 使用 YOLO 找出葉片位置  
            2. 在葉片區域內轉換 HSV 色彩空間  
            3. 計算黃色比例、綠色比例與黃綠比  
            4. 依黃化比例輸出健康、輕度、中度或嚴重黃化診斷  
            """
        )
        return

    all_results: list[AnalysisResult] = []

    if len(uploaded_files) == 1:
        all_results.extend(
            render_single_image_result(
                uploaded_files[0], model, config, conf, iou, max_det, only_largest_leaf
            )
        )
    else:
        st.markdown("### 批次分析結果")
        progress = st.progress(0)
        status = st.empty()

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            status.write(f"正在分析：{uploaded_file.name} ({idx}/{len(uploaded_files)})")
            image = Image.open(uploaded_file)
            image_bgr = pil_to_bgr(image)
            results, _, _ = analyze_image(
                filename=uploaded_file.name,
                image_bgr=image_bgr,
                model=model,
                config=config,
                conf=conf,
                iou=iou,
                max_det=max_det,
                only_largest_leaf=only_largest_leaf,
            )
            all_results.extend(results)
            progress.progress(idx / len(uploaded_files))

        status.success("批次分析完成")

        if all_results:
            df = results_to_dataframe(all_results)
            summary = df.groupby("filename", as_index=False).agg(
                leaf_count=("leaf_id", "count"),
                avg_yellow_ratio=("yellow_ratio_%", "mean"),
                avg_green_ratio=("green_ratio_%", "mean"),
            )
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

            csv_buffer = io.BytesIO(dataframe_to_csv_bytes(df))
            st.download_button(
                "下載完整批次分析 CSV",
                data=csv_buffer,
                file_name="batch_leaf_yellowing_analysis.csv",
                mime="text/csv",
            )
        else:
            st.warning("批次圖片中沒有偵測到葉片。")

    st.markdown("---")
    st.markdown(
        '<div class="small-note">提醒：本系統提供影像輔助診斷，實際田間判斷仍建議搭配水分、土壤、病蟲害與生育期資料。</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()



