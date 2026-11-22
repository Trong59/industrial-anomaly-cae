import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.cm as cm
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.engine.predictor import CAEPredictor, load_cae_from_checkpoint
from src.evaluation.threshold import load_thresholds
from src.utils.config import load_config
from src.utils.visualization import create_anomaly_overlay

# Page configuration
st.set_page_config(
    page_title="AI Visual Inspection - CAE Anomaly Detection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern industrial UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-normal {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        border: 2px solid #31C48D;
    }
    .status-anomaly {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 1.3rem;
        font-weight: bold;
        text-align: center;
        border: 2px solid #F98080;
    }
    .metric-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor_for_category(category: str):
    """Load model and thresholds with Streamlit resource caching for fast inference."""
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models_saved" / category

    ckpt_path = model_dir / "best_model.pth"
    thresh_path = model_dir / f"{category}_thresholds.json"
    config_path = base_dir / "configs" / (f"{category}_e2.yaml" if category == "bottle" else f"{category}.yaml")

    if not config_path.exists():
        config_path = base_dir / "configs" / f"{category}.yaml"

    if not ckpt_path.exists() or not thresh_path.exists():
        return None, None, f"Không tìm thấy file checkpoint hoặc threshold tại: {model_dir}"

    device = torch.device("cpu")
    model, _ = load_cae_from_checkpoint(ckpt_path, device=device)
    thresholds_data = load_thresholds(thresh_path)
    config = load_config(str(config_path))

    data_cfg = config.get("data", {})
    inf_cfg = config.get("inference", {})

    image_thresh = thresholds_data["image_threshold"]
    pixel_thresh = thresholds_data["pixel_threshold"]

    predictor = CAEPredictor(
        model=model,
        device=device,
        image_size=tuple(data_cfg.get("image_size", [256, 256])),
        error_metric=inf_cfg.get("error_metric", "l1"),
        top_percent=inf_cfg.get("top_percent", 0.01),
        image_threshold=image_thresh,
        pixel_threshold=pixel_thresh,
    )

    return predictor, thresholds_data, None


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("Bảng Điều Khiển")
    st.markdown("---")

    # Category selection
    category_options = {
        "bottle": "🍾 Bottle (Chai thủy tinh)",
        "capsule": "💊 Capsule (Viên thuốc nhộng)",
        "metal_nut": "🔩 Metal Nut (Đai ốc kim loại)",
    }

    selected_category = st.selectbox(
        "Chọn loại sản phẩm kiểm tra:",
        options=list(category_options.keys()),
        format_func=lambda x: category_options[x],
        index=0,
    )

    st.markdown("### Thông tin mô hình:")
    predictor, thresholds_data, error_msg = get_predictor_for_category(selected_category)

    if error_msg:
        st.error(error_msg)
    else:
        orig_img_thresh = thresholds_data["image_threshold"]
        orig_px_thresh = thresholds_data["pixel_threshold"]

        st.info(
            f"**Kiến trúc:** CAE + Bottleneck 64 (L1 Loss)\n\n"
            f"**Thiết bị:** CPU (Inference cục bộ)\n\n"
            f"**Ngưỡng ảnh gốc:** `{orig_img_thresh:.6f}`\n\n"
            f"**Ngưỡng pixel gốc:** `{orig_px_thresh:.6f}`"
        )

        # Allow user to fine-tune threshold dynamically if desired
        enable_custom_thresh = st.checkbox("Điều chỉnh ngưỡng thủ công", value=False)
        if enable_custom_thresh:
            custom_thresh = st.slider(
                "Image Threshold:",
                min_value=float(orig_img_thresh * 0.5),
                max_value=float(orig_img_thresh * 2.0),
                value=float(orig_img_thresh),
                step=0.005,
                format="%.4f",
            )
            predictor.image_threshold = custom_thresh
        else:
            predictor.image_threshold = orig_img_thresh

    st.markdown("---")
    st.markdown(
        "**Đề tài nghiên cứu:**\n\n"
        "*Xây dựng và đánh giá mạng Convolutional Autoencoder trong phát hiện và khoanh vùng bất thường bề mặt sản phẩm công nghiệp.*"
    )

# --- MAIN CONTENT AREA ---
st.markdown(
    '<div class="main-header">🏭 HỆ THỐNG PHÁT HIỆN & KHOANH VÙNG KHUYẾT TẬT BỀ MẶT SẢN PHẨM</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="sub-header">Mô hình Convolutional Autoencoder (CAE) — Đang chọn danh mục: <b>{category_options[selected_category]}</b></div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Tải ảnh sản phẩm cần kiểm tra khuyết tật (PNG, JPG, BMP):",
    type=["png", "jpg", "jpeg", "bmp"],
)

if uploaded_file is not None and predictor is not None:
    # Load and process image
    input_image = Image.open(uploaded_file).convert("RGB")

    t_start = time.time()
    result = predictor.predict_single_image(input_image)
    inference_time = (time.time() - t_start) * 1000  # ms

    score = result["image_score"]
    active_thresh = predictor.image_threshold
    is_anomaly = score >= active_thresh

    # --- BANNER RESULT ---
    st.markdown("---")
    if is_anomaly:
        st.markdown(
            f'<div class="status-anomaly">⚠️ KẾT QUẢ: PHÁT HIỆN BẤT THƯỜNG / LỖI BỀ MẶT (ANOMALY)</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="status-normal">✅ KẾT QUẢ: SẢN PHẨM ĐẠT CHUẨN (NORMAL)</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- METRICS SUMMARY CARDS ---
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="Điểm bất thường (Anomaly Score)",
            value=f"{score:.5f}",
            delta=f"{score - active_thresh:+.5f}",
            delta_color="inverse",
        )
    with m2:
        st.metric(label="Ngưỡng phân loại (Threshold)", value=f"{active_thresh:.5f}")
    with m3:
        status_text = "Không đạt ❌" if is_anomaly else "Đạt chuẩn ✔️"
        st.metric(label="Đánh giá chất lượng", value=status_text)
    with m4:
        st.metric(label="Thời gian xử lý", value=f"{inference_time:.1f} ms")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 4-PANEL VISUALIZATION GRID ---
    st.subheader("🖼️ Phân Tích Trực Quan Chi Tiết:")
    col1, col2, col3, col4 = st.columns(4)

    # 1. Original Image
    with col1:
        st.markdown("**1. Ảnh Sản Phẩm Gốc**")
        st.image(result["original"], use_container_width=True)

    # 2. Reconstruction
    with col2:
        st.markdown("**2. Ảnh CAE Tái Tạo**")
        st.image(np.clip(result["reconstruction"], 0, 1), use_container_width=True)

    # 3. Anomaly Heatmap
    with col3:
        st.markdown("**3. Bản Đồ Nhiệt Sai Số (Heatmap)**")
        # Normalize map to [0, 1] for colorful Jet visualization
        amap = result["anomaly_map"]
        norm_map = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        heatmap_colored = cm.jet(norm_map)[:, :, :3]
        st.image(heatmap_colored, use_container_width=True)

    # 4. Defect Overlay & Mask
    with col4:
        st.markdown("**4. Khoanh Vùng Vết Lỗi (Overlay)**")
        pred_mask = result["pred_mask"]
        overlay = create_anomaly_overlay(result["original"], pred_mask, alpha=0.5, color=(1.0, 0.0, 0.0))
        st.image(np.clip(overlay, 0, 1), use_container_width=True)

    # Additional details in expander
    with st.expander("🔍 Xem thông số thống kê chi tiết của ảnh"):
        st.write(
            {
                "Kích thước tensor đầu vào": list(result["original"].shape),
                "Sai số pixel nhỏ nhất (Min)": float(amap.min()),
                "Sai số pixel trung bình (Mean)": float(amap.mean()),
                "Sai số pixel lớn nhất (Max)": float(amap.max()),
                "Số điểm ảnh vượt ngưỡng lỗi (Mask Pixels)": int(np.sum(pred_mask > 0.5)),
                "Thiết bị thực thi": "Intel/AMD CPU (Local Inference)",
            }
        )

else:
    # Instructions when no image is uploaded
    st.info("💡 **Hướng dẫn:** Chọn loại sản phẩm ở bảng bên trái, sau đó kéo-thả hoặc tải một bức ảnh lên để hệ thống tự động kiểm tra!")
