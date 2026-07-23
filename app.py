# -*- coding: utf-8 -*-
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
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - Streamlit shows a friendly error at runtime.
    YOLO = None

APP_TITLE = "大豆葉片黃化程度分析平台"
DEFAULT_WEIGHT_PATH = "weights/best.pt"
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp"]
MAX_IMAGE_SIDE = 1920

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .hero { border: 1px solid #dbeafe; border-radius: 26px; padding: 28px 30px; margin-bottom: 20px; background: linear-gradient(135deg, #ecfdf5 0%, #eff6ff 55%, #fff7ed 100%); box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08); }
        .main-title { font-size: 2.35rem; font-weight: 900; margin-bottom: 0.35rem; color: #0f172a; }
        .sub-title { color: #475569; font-size: 1.08rem; margin-bottom: 1.2rem; line-height: 1.7; }
        .badge-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 14px; }
        .badge { display: inline-flex; align-items: center; gap: 6px; border: 1px solid #bbf7d0; background: rgba(240, 253, 244, 0.85); color: #166534; border-radius: 999px; padding: 7px 12px; font-size: 0.9rem; font-weight: 700; }
        .feature-card { border: 1px solid #e2e8f0; border-radius: 18px; padding: 18px; height: 100%; background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%); }
        .feature-title { font-weight: 850; color: #0f172a; font-size: 1.02rem; margin-bottom: 6px; }
        .feature-text { color: #64748b; font-size: 0.92rem; line-height: 1.55; }
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
        .pipeline { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-top: 10px; }
        .pipe-step { border: 1px solid #dbeafe; border-radius: 16px; padding: 14px 12px; background: #eff6ff; text-align: center; color: #1e3a8a; font-weight: 750; font-size: 0.92rem; }
        .interpretation { border-left: 5px solid #22c55e; background: #f0fdf4; border-radius: 14px; padding: 14px 16px; color: #14532d; line-height: 1.65; margin: 12px 0; }
        .limit-box { border: 1px solid #fed7aa; background: #fff7ed; color: #9a3412; border-radius: 16px; padding: 15px 16px; line-height: 1.65; }
        @media (max-width: 900px) { .pipeline { grid-template-columns: 1fr; } .main-title { font-size: 1.75rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_yolo_model(weight_path: str) -> Any | None:
    """Load YOLO once per weight path to avoid reloading on every Streamlit rerun."""
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


def safe_st_image(image: Any, caption: str) -> None:
    """Compatible image rendering for both old and new Streamlit versions."""
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image, caption=caption, use_column_width=True)


def load_uploaded_image(uploaded_file: Any) -> Image.Image | None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        st.error(f"無法讀取圖片 {uploaded_file.name}：{exc}")
        return None


def pil_to_bgr(image: Image.Image, resize_long_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return resize_if_needed(bgr, resize_long_side)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_if_needed(image_bgr: np.ndarray, long_side: int) -> np.ndarray:
    if long_side <= 0:
        return image_bgr
    height, width = image_bgr.shape[:2]
    current_long_side = max(height, width)
    if current_long_side <= long_side:
        return image_bgr
    scale = long_side / current_long_side
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)


def apply_gray_world_white_balance(image_bgr: np.ndarray) -> np.ndarray:
    """Simple gray-world white balance for field photos with color cast."""
    image = image_bgr.astype(np.float32)
    channel_means = image.reshape(-1, 3).mean(axis=0)
    gray_mean = channel_means.mean()
    scale = gray_mean / np.maximum(channel_means, 1e-6)
    balanced = np.clip(image * scale, 0, 255).astype(np.uint8)
    return balanced


def apply_clahe_to_value_channel(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def preprocess_image(image_bgr: np.ndarray, config: RuntimeConfig) -> np.ndarray:
    processed = image_bgr
    if config.apply_gray_world:
        processed = apply_gray_world_white_balance(processed)
    if config.apply_clahe:
        processed = apply_clahe_to_value_channel(processed)
    return processed


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def get_leaf_detections(model: Any, image_bgr: np.ndarray, conf: float, iou: float, max_det: int) -> list[dict[str, Any]]:
    results = model.predict(image_bgr, conf=conf, iou=iou, max_det=max_det, verbose=False)
    if not results or results[0].boxes is None:
        return []

    result = results[0]
    detections: list[dict[str, Any]] = []
    height, width = image_bgr.shape[:2]

    for box in result.boxes:
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


def build_hsv_config() -> HSVConfig:
    st.sidebar.markdown("### 黃化判斷參數 HSV")
    st.sidebar.caption("HSV 參數會影響哪些像素被視為黃色或綠色。若照片偏黃、偏暗或光線不均，可小幅調整。")

    with st.sidebar.expander("黃色像素判斷範圍", expanded=False):
        st.caption("調整哪些顏色會被計入黃化區域。範圍放寬會提高黃化比例，範圍縮小會讓判斷較嚴格。")
        y_h_min = st.slider("黃色色相下限", 0, 179, 18, help="Hue 下限。預設提高到 18，避免把偏橘背景誤算為黃化。")
        y_h_max = st.slider("黃色色相上限", 0, 179, 38, help="Hue 上限。預設收斂到 38，讓黃化判斷更嚴格。")
        y_s_min = st.slider("黃色最低飽和度", 0, 255, 60, help="排除灰白或低彩度背景。數值越高，黃色判斷越嚴格。")
        y_v_min = st.slider("黃色最低亮度", 0, 255, 80, help="排除過暗像素。數值越高，暗部越不容易被算作黃化。")

    with st.sidebar.expander("綠色像素判斷範圍", expanded=False):
        st.caption("用來估計健康綠色區域。綠色比例可作為黃化比例的輔助對照。")
        g_h_min = st.slider("綠色色相下限", 0, 179, 35)
        g_h_max = st.slider("綠色色相上限", 0, 179, 90)
        g_s_min = st.slider("綠色最低飽和度", 0, 255, 30)
        g_v_min = st.slider("綠色最低亮度", 0, 255, 45)

    with st.sidebar.expander("有效葉片像素過濾", expanded=False):
        st.caption("用來排除太暗、太灰或背景區域。預設值已提高，以減少背景被誤判為葉片；調太高可能漏掉陰影中的葉片。")
        min_sat = st.slider("有效像素最低飽和度", 0, 255, 40)
        min_val = st.slider("有效像素最低亮度", 0, 255, 45)
        morph_kernel_size = st.slider("遮罩去雜訊強度", 0, 9, 3, 2, help="用形態學開閉運算移除零星雜點。0 代表不處理。")

    return HSVConfig(
        yellow_lower=(y_h_min, y_s_min, y_v_min),
        yellow_upper=(y_h_max, 255, 255),
        green_lower=(g_h_min, g_s_min, g_v_min),
        green_upper=(g_h_max, 255, 255),
        min_leaf_saturation=min_sat,
        min_leaf_value=min_val,
        morph_kernel_size=morph_kernel_size,
    )


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

    yellow_ratio = yellow_area / leaf_area * 100
    green_ratio = green_area / leaf_area * 100
    yellow_green_ratio = yellow_area / max(green_area, 1)
    return yellow_ratio, green_ratio, yellow_green_ratio, leaf_area, yellow_mask


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


def analyze_image(
    filename: str,
    image_bgr: np.ndarray,
    model: Any | None,
    config: RuntimeConfig,
) -> tuple[list[AnalysisResult], np.ndarray, list[np.ndarray]]:
    processed_bgr = preprocess_image(image_bgr, config)

    if model is None:
        h, w = processed_bgr.shape[:2]
        detections = [{"bbox": (0, 0, w, h), "confidence": 1.0, "class_id": 0, "area": w * h}]
    else:
        detections = get_leaf_detections(model, processed_bgr, conf=config.conf, iou=config.iou, max_det=config.max_det)

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

    annotated = draw_detection_summary(processed_bgr, results)
    return results, annotated, overlays


def results_to_dataframe(results: list[AnalysisResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
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
        ]
    )


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def render_header() -> None:
    st.markdown(
        f'''
        <div class="hero">
            <div class="main-title">🌱 {APP_TITLE}</div>
            <div class="sub-title">以 YOLO11 葉片偵測結合 HSV 色彩空間分析，將大豆葉片影像轉換為可解釋的黃化比例、分級結果與 CSV 數據，適合農業研究、教學展示與田間初步篩檢。</div>
            <div class="badge-row"><span class="badge">YOLO11n 偵測</span><span class="badge">HSV 黃化量化</span><span class="badge">批次 CSV 匯出</span><span class="badge">HPC 訓練成果</span></div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_platform_overview() -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="feature-card"><div class="feature-title">① 自動找出葉片</div><div class="feature-text">使用 YOLO 模型定位葉片區域，降低背景對黃化比例估算的干擾。</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="feature-card"><div class="feature-title">② 量化黃化程度</div><div class="feature-text">於 HSV 色彩空間計算黃色、綠色與黃綠比，輸出可比較的數值指標。</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="feature-card"><div class="feature-title">③ 產生研究資料</div><div class="feature-text">支援單張與批次分析，可下載 CSV，方便放入報告與後續統計。</div></div>', unsafe_allow_html=True)


def render_model_summary() -> None:
    st.markdown('### 模型與訓練成果')
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric('模型', 'YOLO11n')
    with c2:
        render_metric('mAP50', '0.995')
    with c3:
        render_metric('mAP50-95', '0.980')
    with c4:
        render_metric('HPC 訓練', '50 epochs')
    st.caption('以上為 w3_formal_baseline_hpc baseline 訓練結果；本平台以此作為葉片偵測與黃化量化展示基礎。')


def render_pipeline() -> None:
    st.markdown('### 分析流程')
    st.markdown('''<div class="pipeline"><div class="pipe-step">上傳圖片</div><div class="pipe-step">YOLO 偵測</div><div class="pipe-step">葉片 ROI</div><div class="pipe-step">HSV 分析</div><div class="pipe-step">分級與匯出</div></div>''', unsafe_allow_html=True)


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


def render_sidebar() -> RuntimeConfig:
    st.sidebar.title("⚙️ 分析參數設定")
    st.sidebar.caption("這些設定會影響葉片偵測與黃化比例計算；若不確定，建議使用預設值。")

    weight_path = st.sidebar.text_input("YOLO 權重路徑", DEFAULT_WEIGHT_PATH)
    st.sidebar.caption("建議將訓練好的模型放在 weights/best.pt")

    st.sidebar.markdown("### YOLO 葉片偵測設定")
    st.sidebar.caption("控制模型要多嚴格地判斷葉片位置。信心值越高，誤偵測較少，但可能漏掉較不清楚的葉片。")
    conf = st.sidebar.slider("葉片偵測信心值", 0.05, 0.95, 0.25, 0.05, help="數值越高，模型越保守；數值越低，較容易偵測到葉片，但也可能增加誤偵測。")
    iou = st.sidebar.slider("重疊框合併門檻 IoU", 0.10, 0.90, 0.45, 0.05, help="用來合併重疊的偵測框。一般使用預設值即可。")
    max_det = st.sidebar.slider("最多偵測葉片數", 1, 50, 10, 1, help="限制單張圖片最多分析幾片葉片，避免背景誤判造成過多結果。")
    only_largest_leaf = st.sidebar.checkbox("只分析最大葉片", value=False, help="適合單片葉片照片；若圖片有多片葉片，建議取消勾選。")

    st.sidebar.markdown("### 影像前處理")
    resize_long_side = st.sidebar.slider("最大影像邊長", 640, 2560, MAX_IMAGE_SIDE, 160, help="上傳大圖時先縮小，可降低記憶體用量並加快分析。")
    apply_gray_world = st.sidebar.checkbox("自動白平衡", value=True, help="校正田間照片偏黃、偏藍等色偏；若使用標準色卡拍攝可關閉。")
    apply_clahe = st.sidebar.checkbox("亮度對比增強", value=False, help="改善陰影或逆光照片；若黃化比例被放大，可關閉。")

    hsv_config = build_hsv_config()

    st.sidebar.markdown("---")
    st.sidebar.info("若模型權重不存在，平台會暫時以整張圖片進行 HSV 分析，僅供測試介面，不能視為正式葉片偵測結果。")
    return RuntimeConfig(
        weight_path=weight_path,
        conf=conf,
        iou=iou,
        max_det=max_det,
        only_largest_leaf=only_largest_leaf,
        resize_long_side=resize_long_side,
        apply_gray_world=apply_gray_world,
        apply_clahe=apply_clahe,
        hsv=hsv_config,
    )


def render_single_image_result(uploaded_file: Any, model: Any | None, config: RuntimeConfig) -> list[AnalysisResult]:
    image = load_uploaded_image(uploaded_file)
    if image is None:
        return []

    image_bgr = pil_to_bgr(image, config.resize_long_side)

    start = time.time()
    results, annotated, overlays = analyze_image(uploaded_file.name, image_bgr, model, config)
    elapsed = time.time() - start

    if not results:
        st.warning("沒有偵測到葉片。可以降低 confidence，或確認圖片中葉片是否清楚。")
        safe_st_image(image, caption="原始圖片")
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

    st.markdown(
        f"""
        <div class="diagnosis-box {diag_class}">
            <h3 style="margin: 0 0 6px 0;">黃化分級：{main_diagnosis}</h3>
            <div>{suggestion}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'''
        <div class="interpretation">
            本張影像平均黃化比例為 <b>{avg_yellow:.1f}%</b>，平均綠色比例為 <b>{avg_green:.1f}%</b>。
            系統依黃化比例判定為「<b>{main_diagnosis}</b>」。此結果可作為葉片黃化程度量化參考，後續仍建議搭配田間水分、土壤養分與病蟲害紀錄判讀。
        </div>
        ''',
        unsafe_allow_html=True,
    )

    if model is None:
        st.markdown(
            '<div class="warn-note">目前未載入 YOLO 權重，因此結果是「整張圖片 HSV 測試分析」，不是正式葉片偵測分析。</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### 影像分析結果")
    col1, col2, col3 = st.columns(3)
    with col1:
        safe_st_image(image, caption="原始圖片")
    with col2:
        safe_st_image(bgr_to_rgb(annotated), caption="YOLO 偵測與黃化比例")
    with col3:
        if overlays:
            safe_st_image(bgr_to_rgb(overlays[0]), caption="葉片 ROI 黃化遮罩（已降低背景干擾）")
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


def render_batch_results(uploaded_files: list[Any], model: Any | None, config: RuntimeConfig) -> list[AnalysisResult]:
    st.markdown("### 批次分析結果")
    progress = st.progress(0)
    status = st.empty()
    all_results: list[AnalysisResult] = []

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        status.write(f"正在分析：{uploaded_file.name} ({idx}/{len(uploaded_files)})")
        image = load_uploaded_image(uploaded_file)
        if image is None:
            progress.progress(idx / len(uploaded_files))
            continue
        image_bgr = pil_to_bgr(image, config.resize_long_side)
        results, _, _ = analyze_image(uploaded_file.name, image_bgr, model, config)
        all_results.extend(results)
        progress.progress(idx / len(uploaded_files))

    status.success("批次分析完成")

    if not all_results:
        st.warning("批次圖片中沒有偵測到葉片。")
        return []

    df = results_to_dataframe(all_results)
    summary = df.groupby("filename", as_index=False).agg(
        leaf_count=("leaf_id", "count"),
        avg_yellow_ratio=("yellow_ratio_%", "mean"),
        avg_green_ratio=("green_ratio_%", "mean"),
    )
    summary["avg_yellow_ratio"] = summary["avg_yellow_ratio"].round(2)
    summary["avg_green_ratio"] = summary["avg_green_ratio"].round(2)
    summary["diagnosis"] = summary["avg_yellow_ratio"].apply(lambda x: diagnose(float(x))[0])

    diagnosis_counts = summary["diagnosis"].value_counts().to_dict()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric("圖片數", str(len(uploaded_files)))
    with c2:
        render_metric("總葉片數", str(len(all_results)))
    with c3:
        render_metric("整體平均黃化", f"{summary['avg_yellow_ratio'].mean():.1f}%")
    with c4:
        render_metric("需關注圖片", str(int((summary["avg_yellow_ratio"] >= 25).sum())))

    st.markdown(
        f'''<div class="interpretation">批次分析共處理 <b>{len(uploaded_files)}</b> 張圖片、<b>{len(all_results)}</b> 個葉片偵測結果；其中中度以上黃化圖片數為 <b>{int((summary['avg_yellow_ratio'] >= 25).sum())}</b> 張。分級分布：{diagnosis_counts}</div>''',
        unsafe_allow_html=True,
    )

    st.markdown("#### 每張圖片摘要")
    st.dataframe(summary, use_container_width=True)
    st.markdown("#### 每片葉片詳細資料")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "下載完整批次分析 CSV",
        data=io.BytesIO(dataframe_to_csv_bytes(df)),
        file_name="batch_leaf_yellowing_analysis.csv",
        mime="text/csv",
    )
    return all_results


def render_empty_state() -> None:
    st.info("請上傳一張或多張葉片圖片開始分析。")
    render_model_summary()
    render_pipeline()
    st.markdown(
        """
        ### 拍攝建議
        - 盡量讓葉片清楚、不要嚴重模糊。
        - 避免強烈反光、過暗陰影或背景顏色太接近葉片。
        - 單片葉片照片可勾選「只分析最大葉片」；多片葉片照片則建議取消勾選。
        - 系統輸出的是黃化程度量化，不直接判定病因。
        """
    )


def main() -> None:
    inject_css()
    render_header()
    render_platform_overview()
    config = render_sidebar()
    model = load_yolo_model(config.weight_path)

    st.markdown("### 上傳葉片圖片")
    st.caption("適用於農業研究、教學展示、作物影像分析與田間初步篩檢；本平台提供黃化比例量化，不直接判定病因。")
    uploaded_files = st.file_uploader(
        "支援 JPG / JPEG / PNG / BMP，可一次上傳多張進行批次分析。",
        type=SUPPORTED_IMAGE_TYPES,
        accept_multiple_files=True,
    )

    if not uploaded_files:
        render_empty_state()
    elif len(uploaded_files) == 1:
        render_single_image_result(uploaded_files[0], model, config)
    else:
        render_batch_results(list(uploaded_files), model, config)

    st.markdown("---")
    st.markdown(
        '<div class="limit-box"><b>研究限制與適用範圍：</b>本系統提供黃化比例量化與影像輔助判斷，不等同完整病害診斷；實際田間判斷仍建議搭配水分、土壤、病蟲害、生育期與環境資料。若照片過暗、過曝、葉片被遮蔽或背景干擾嚴重，黃化比例可能產生偏差。</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
