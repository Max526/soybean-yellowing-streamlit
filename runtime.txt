# 大豆葉片黃化程度分析 Streamlit 平台

這是一個以 YOLO 葉片偵測與 HSV 色彩分析為核心的 Streamlit 平台，用於量化大豆葉片黃化比例。

本平台定位為「黃化程度量化工具」，適合農業研究、教學展示、作物影像分析與田間初步篩檢；輸出結果可輔助觀察黃化程度，但不等同完整病害診斷。

## 本版修正重點

- 修正 Streamlit Cloud apt 階段的 `packages.txt` 錯誤：本專案使用 `opencv-python-headless`，不需要額外安裝 `libgl1` / `libglib2.0-0t64`，因此將 `packages.txt` 清空，避免 Cloud 解析到錯誤內容後出現 `xargs: unmatched double quote` 或 `Command line option '>' is not understood`。
- 修正 Streamlit 新舊版本 `st.image` 參數相容性，避免 `use_container_width` / `use_column_width` 部署錯誤。
- 圖片讀取加入 EXIF 方向修正，手機拍攝照片不易旋轉錯誤。
- 大圖自動縮小到指定最大邊長，降低 Streamlit Cloud 記憶體壓力並加快推論。
- 加入自動白平衡與亮度對比增強選項，改善田間照片色偏與陰影問題。
- 黃化遮罩加入形態學去雜訊，減少零星背景像素干擾。
- 將參數集中成 `RuntimeConfig` / `HSVConfig`，讓後續維護與實驗更清楚。
- 單張與批次流程拆分成獨立函式，降低主程式複雜度。
- 模型缺失時明確提示「整張圖片 HSV 測試分析」，避免被誤認為正式 YOLO 偵測結果。

## 線上部署建議

本專案適合部署到 Streamlit Community Cloud。

主程式：

```text
app.py
```

部署時需要的檔案：

```text
app.py
requirements.txt
runtime.txt
packages.txt
.streamlit/config.toml
README.md
```

`packages.txt` 目前刻意保持空白，因為 `requirements.txt` 使用的是 `opencv-python-headless`，部署時不需要額外 apt 套件。請不要把錯誤訊息、shell 指令或 Markdown 文字貼進 `packages.txt`。

若使用 YOLO 權重，建議另外放入：

```text
weights/best.pt
```

如果 `best.pt` 太大，不建議直接放 GitHub。可改放 Hugging Face Hub、Google Drive 或 GitHub Release，再於程式啟動時下載。

## 功能

- 支援 JPG、JPEG、PNG、BMP 影像上傳
- 可上傳單張或多張葉片圖片
- YOLO 偵測葉片位置
- HSV 分析黃色比例、綠色比例與黃綠比
- 輸出健康、輕度黃化、中度黃化、嚴重黃化分級
- 顯示原圖、偵測圖與黃化遮罩
- 支援批次分析與 CSV 匯出
- 側邊欄提供白話化參數說明
- 可調整 YOLO confidence、IoU、影像前處理與 HSV 閾值

## 介面設計重點

側邊欄已改為「分析參數設定」，並加入每個參數的用途說明：

- 葉片偵測信心值：控制 YOLO 偵測的嚴格程度
- 重疊框合併門檻 IoU：控制重疊偵測框的合併方式
- 最多偵測葉片數：限制單張圖片分析葉片數量
- 最大影像邊長：控制分析前縮圖尺寸，降低記憶體用量
- 自動白平衡：降低照片色偏對黃化比例的影響
- 亮度對比增強：改善陰影與逆光，但可能放大部分黃化比例
- 黃色像素判斷範圍：決定哪些像素被視為黃化區域
- 綠色像素判斷範圍：估計健康綠色區域比例
- 有效葉片像素過濾：排除過暗、過灰或背景區域
- 遮罩去雜訊強度：移除零星雜點，讓黃化遮罩更穩定

若使用者不熟悉影像處理，建議保留預設值。

## 模型權重

預設會讀取：

```text
weights/best.pt
```

如果沒有放模型，平台會暫時以整張圖片做 HSV 分析，方便先測試介面。這種模式不能視為真正的 YOLO 葉片偵測結果。

## 安裝套件

```bash
pip install -r requirements.txt
```

## 本機啟動

```bash
streamlit run app.py
```

啟動後開啟：

```text
http://127.0.0.1:8501
```

## Streamlit Community Cloud 部署步驟

1. 將本專案上傳到 GitHub repository。
2. 進入 https://share.streamlit.io/ 。
3. 使用 GitHub 帳號登入。
4. 點選 New app。
5. Repository 選擇你的專案。
6. Branch 選擇 main。
7. Main file path 輸入：

```text
app.py
```

8. 點 Deploy。

## 診斷規則

| 黃化比例 | 分級 |
|---|---|
| < 10% | 健康 / 正常 |
| 10–25% | 輕度黃化 |
| 25–45% | 中度黃化 |
| >= 45% | 嚴重黃化 |

## 注意事項

本平台提供黃化比例量化與影像輔助判斷，不等同完整病害診斷。若要作為正式田間判斷，建議搭配：

- 土壤含水量
- 淹水處理時間
- 植株生育期
- 根系狀態
- 病蟲害紀錄
- 人工標註驗證資料
- 固定拍攝距離、光源與背景條件

## 後續可優化方向

- 加入標準色卡校正
- 加入使用者自訂黃化分級門檻
- 加入多葉片個別分析 PDF / HTML 報告
- 依實際田間資料校準黃化分級門檻
- 補入 YOLO 評估指標，例如 Precision、Recall、mAP50、mAP50-95
- 若未來有 segmentation 權重，可改用 YOLO11-Seg 遮罩取代 bounding box ROI，提高葉片區域準確度
