from pathlib import Path
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="大豆葉片黃化線上輔助診斷平台", page_icon="🌱", layout="wide")

HSV_PRESETS = {
    "保守 Conservative（低誤判）": ((22, 60, 100), (35, 255, 255)),
    "平衡 Balanced（建議預設）": ((18, 40, 80), (42, 255, 255)),
    "寬鬆 Loose（抓淡黃）": ((15, 35, 70), (45, 255, 255)),
}
LOWER_LEAF = (15, 25, 40)
UPPER_LEAF = (95, 255, 255)
DEFAULT_WEIGHTS = Path("weights/best.pt")

@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    return YOLO(weights_path)

def diagnosis(ratio: float):
    pct = ratio * 100
    if pct < 10:
        return "正常 / 低黃化", "success", "葉色大致正常。建議持續觀察植株生長狀況。"
    if pct < 30:
        return "輕微黃化", "info", "已有輕微黃化現象。建議觀察新葉、澆水狀況與近期施肥紀錄。"
    if pct < 60:
        return "中度黃化", "warning", "黃化程度中等。可能與缺氮、根部壓力、淹水或病蟲害有關，建議檢查土壤水分與養分。"
    return "嚴重黃化", "error", "黃化比例偏高。建議盡快檢查排水、根系、土壤養分與病害風險，必要時請農業專業人員協助判斷。"

def put_label(img, text, x, y, color=(0, 0, 255)):
    width = min(img.shape[1] - 1, x + max(430, len(text) * 11))
    cv2.rectangle(img, (x, max(0, y - 26)), (width, min(img.shape[0] - 1, y + 5)), (255, 255, 255), -1)
    cv2.putText(img, text, (x + 4, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

def analyze_image(image_rgb, model, conf, iou, lower_yellow, upper_yellow, use_lab=True, lab_b_min=135):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, image_bgr)
        result = model.predict(source=tmp.name, conf=conf, iou=iou, verbose=False)[0]

    annotated = image_bgr.copy()
    yellow_canvas = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    total_yellow = 0
    total_leaf = 0
    detections = []

    if result.boxes is not None and len(result.boxes) > 0:
        for idx, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_bgr.shape[1] - 1, x2), min(image_bgr.shape[0] - 1, y2)
            score = float(box.conf[0])
            cls_id = int(box.cls[0]) if box.cls is not None else 0
            cls_name = result.names.get(cls_id, str(cls_id))
            roi = image_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            yellow_mask = cv2.inRange(hsv, np.array(lower_yellow), np.array(upper_yellow))
            leaf_mask = cv2.inRange(hsv, np.array(LOWER_LEAF), np.array(UPPER_LEAF))

            if use_lab:
                lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
                lab_b = lab[:, :, 2]
                lab_yellow = (lab_b >= lab_b_min).astype(np.uint8) * 255
                yellow_mask = cv2.bitwise_and(yellow_mask, lab_yellow)

            kernel = np.ones((3, 3), np.uint8)
            yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
            leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

            yellow_count = int(np.count_nonzero(yellow_mask))
            leaf_count = int(np.count_nonzero(leaf_mask))
            local_ratio = yellow_count / leaf_count if leaf_count else 0.0
            total_yellow += yellow_count
            total_leaf += leaf_count
            yellow_canvas[y1:y2, x1:x2] = np.maximum(yellow_canvas[y1:y2, x1:x2], yellow_mask)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 0), 3)
            put_label(annotated, f"{cls_name} {score:.2f} | yellow {local_ratio*100:.1f}%", x1, max(28, y1))
            detections.append({
                "編號": idx + 1,
                "類別": cls_name,
                "信心分數": round(score, 4),
                "黃化比例(%)": round(local_ratio * 100, 2),
                "有效葉片像素": leaf_count,
                "黃色像素": yellow_count,
                "bbox": [x1, y1, x2, y2],
            })

    total_ratio = total_yellow / total_leaf if total_leaf else 0.0
    diag, level, advice = diagnosis(total_ratio)
    overlay = annotated.copy()
    overlay[yellow_canvas > 0] = (0, 165, 255)
    annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)
    put_label(annotated, f"Yellow ratio: {total_ratio*100:.1f}% | {diag}", 10, 32)
    mask_rgb = np.zeros((*yellow_canvas.shape, 3), dtype=np.uint8)
    mask_rgb[yellow_canvas > 0] = (255, 190, 0)
    return {
        "annotated_rgb": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        "yellow_mask_rgb": mask_rgb,
        "detections": detections,
        "total_yellow_ratio": total_ratio,
        "diagnosis": diag,
        "level": level,
        "advice": advice,
        "total_leaf_pixels": total_leaf,
        "total_yellow_pixels": total_yellow,
    }

def show_diagnosis_box(level, text):
    if level == "success": st.success(text)
    elif level == "info": st.info(text)
    elif level == "warning": st.warning(text)
    else: st.error(text)

def result_badge(level):
    return {"success": "🟢", "info": "🔵", "warning": "🟠", "error": "🔴"}.get(level, "⚪")

def render_result(name, image_rgb, result):
    ratio_pct = result["total_yellow_ratio"] * 100
    st.subheader(f"診斷結果：{name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("偵測到的大豆葉片", len(result["detections"]))
    c2.metric("黃化比例", f"{ratio_pct:.2f}%")
    c3.metric("診斷等級", result["diagnosis"])
    c4.metric("有效葉片像素", f"{result['total_leaf_pixels']:,}")
    show_diagnosis_box(result["level"], f"{result_badge(result['level'])} {result['diagnosis']}｜{result['advice']}")

    left, right = st.columns(2)
    with left:
        st.image(image_rgb, caption="原始照片", use_container_width=True)
    with right:
        st.image(result["annotated_rgb"], caption="YOLO 葉片定位 + 黃化遮罩", use_container_width=True)
    st.image(result["yellow_mask_rgb"], caption="系統判定的黃色 / 黃化區域 mask", use_container_width=True)

    with st.expander("查看每個葉片框的詳細數值"):
        if result["detections"]:
            st.dataframe(result["detections"], use_container_width=True)
        else:
            st.warning("沒有偵測到 leaf。可嘗試降低 confidence、換更清楚的大豆葉片照片，或確認權重是否正確。")

st.title("🌱 大豆葉片黃化偵測與線上輔助診斷平台")
st.markdown("""
本系統以 **YOLO 偵測大豆葉片位置**，再使用 **HSV + LAB 色彩分析** 計算黃化比例，提供正常、輕微、中度、嚴重黃化的輔助診斷。
""")

st.info("操作流程：上傳大豆葉片照片 → YOLO 找葉片 → 色彩分析計算黃化比例 → 顯示診斷等級與建議。")

with st.sidebar:
    st.header("大豆黃化分析設定")
    weights_path = st.text_input("YOLO 權重", value=str(DEFAULT_WEIGHTS))
    preset_name = st.selectbox("黃化偵測靈敏度", list(HSV_PRESETS.keys()), index=1)
    compare_presets = st.checkbox("顯示三種靈敏度比較", value=True)
    lower_yellow, upper_yellow = HSV_PRESETS[preset_name]

    with st.expander("進階參數", expanded=False):
        conf = st.slider("YOLO confidence", 0.05, 0.95, 0.25, 0.05)
        iou = st.slider("YOLO IoU", 0.1, 0.9, 0.70, 0.05)
        use_lab = st.checkbox("啟用 LAB b 通道輔助過濾", value=True)
        lab_b_min = st.slider("LAB b 最低值", 120, 170, 135, 5, disabled=not use_lab)
        st.code(f"HSV: {lower_yellow} → {upper_yellow}\nLAB b >= {lab_b_min if use_lab else '未啟用'}")
    st.caption("建議一般展示使用 Balanced；背景誤判多用 Conservative；淡黃抓不到用 Loose。")

if not Path(weights_path).exists():
    st.error(f"找不到 YOLO 權重：{weights_path}")
    st.stop()

with st.spinner("正在載入 YOLO 模型..."):
    model = load_model(weights_path)

tab_single, tab_batch, tab_method = st.tabs(["單張診斷", "批次分析", "方法說明"])

with tab_single:
    uploaded = st.file_uploader("上傳一張大豆葉片照片", type=["jpg", "jpeg", "png", "webp"], key="single")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_rgb = np.array(image)

        st.markdown("### 分析範圍")
        st.warning("你這張圖的 YOLO 框到整個盆栽/場景，黃化葉片會被大量正常葉片與背景稀釋，所以比例只有幾%。請啟用手動 ROI，只框住要診斷的大豆葉片。")
        use_manual_roi = st.checkbox("啟用手動 ROI，只分析指定葉片區域", value=False)
        analysis_rgb = image_rgb
        if use_manual_roi:
            h, w = image_rgb.shape[:2]
            c1, c2 = st.columns(2)
            with c1:
                x1 = st.slider("ROI 左邊界 x1", 0, w - 1, max(0, int(w * 0.35)))
                x2 = st.slider("ROI 右邊界 x2", 1, w, min(w, int(w * 0.75)))
            with c2:
                y1 = st.slider("ROI 上邊界 y1", 0, h - 1, max(0, int(h * 0.10)))
                y2 = st.slider("ROI 下邊界 y2", 1, h, min(h, int(h * 0.75)))
            if x2 <= x1 or y2 <= y1:
                st.error("ROI 範圍不正確，請讓右/下邊界大於左/上邊界。")
                st.stop()
            preview = image_rgb.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 0), 4)
            st.image(preview, caption="手動 ROI 預覽：藍框內才會進行黃化分析", use_container_width=True)
            analysis_rgb = image_rgb[y1:y2, x1:x2]

        with st.spinner("正在分析照片..."):
            result = analyze_image(analysis_rgb, model, conf, iou, lower_yellow, upper_yellow, use_lab, lab_b_min)
        render_result(uploaded.name + ("（手動 ROI）" if use_manual_roi else ""), analysis_rgb, result)

        if compare_presets:
            st.divider()
            st.subheader("黃化靈敏度比較")
            cols = st.columns(3)
            for col, (pname, (lo, hi)) in zip(cols, HSV_PRESETS.items()):
                r = analyze_image(analysis_rgb, model, conf, iou, lo, hi, use_lab, lab_b_min)
                with col:
                    st.metric(pname.split("（")[0], f"{r['total_yellow_ratio']*100:.2f}%")
                    st.caption(r["diagnosis"])
                    st.image(r["yellow_mask_rgb"], use_container_width=True)
    else:
        st.info("請上傳一張大豆葉片照片開始診斷。")

with tab_batch:
    uploads = st.file_uploader("上傳多張大豆葉片照片", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key="batch")
    if uploads:
        rows = []
        for uploaded in uploads:
            image = Image.open(uploaded).convert("RGB")
            image_rgb = np.array(image)
            result = analyze_image(image_rgb, model, conf, iou, lower_yellow, upper_yellow, use_lab, lab_b_min)
            rows.append({
                "圖片": uploaded.name,
                "偵測葉片數": len(result["detections"]),
                "黃化比例(%)": round(result["total_yellow_ratio"] * 100, 2),
                "診斷": result["diagnosis"],
                "黃色像素": result["total_yellow_pixels"],
                "有效葉片像素": result["total_leaf_pixels"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.download_button("下載批次分析 CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name="soybean_yellowing_results.csv", mime="text/csv")
        st.bar_chart(df.set_index("圖片")[["黃化比例(%)"]])
    else:
        st.info("可一次上傳多張照片，快速比較不同植株或不同處理組的大豆黃化程度。")

with tab_method:
    st.subheader("系統判斷邏輯")
    st.markdown("""
1. **YOLO 葉片定位**：先找出照片中的大豆葉片 / 植株區域。  
2. **有效葉片像素篩選**：在 YOLO 框內排除部分背景、陰影與低飽和區域。  
3. **HSV 黃色區域偵測**：找出偏黃或黃綠色像素。  
4. **LAB b 通道輔助過濾**：降低反光、灰白區域被誤判成黃化的機率。  
5. **黃化比例計算**：黃色像素 ÷ 有效葉片像素。  
6. **輔助診斷分級**：依黃化比例分為正常、輕微、中度、嚴重。
""")
    st.table(pd.DataFrame([
        {"黃化比例": "< 10%", "診斷": "正常 / 低黃化", "說明": "葉色大致正常，持續觀察"},
        {"黃化比例": "10–30%", "診斷": "輕微黃化", "說明": "觀察新葉、澆水與施肥紀錄"},
        {"黃化比例": "30–60%", "診斷": "中度黃化", "說明": "檢查缺氮、淹水、根部壓力或病蟲害"},
        {"黃化比例": "> 60%", "診斷": "嚴重黃化", "說明": "建議盡快檢查栽培環境與植株健康"},
    ]))
    st.warning("本平台為輔助診斷工具，結果需搭配田間觀察、土壤水分、養分狀態與農業專業判斷。")

st.caption("專題定位：基於 YOLO 的大豆葉片黃化偵測與線上輔助診斷平台。")
