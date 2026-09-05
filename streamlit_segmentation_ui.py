from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from config import CHECKPOINT_PATH, IGNORE_INDEX, NUM_CLASSES, VOC_ROOT
from dataset_voc import read_split_ids
from inference import (
    load_checkpoint_model,
    overlay_mask,
    predict_original_size,
    predict_with_uncertainty,
)
from plot_training_curves import create_training_figure
from voc_meta import VOC_CLASSES, mask_to_color_rgb

MAX_PIXELS = 20_000_000  # Giới hạn 20 Megapixels tránh cạn kiệt bộ nhớ


def summarize_present_classes(mask: np.ndarray, min_area_percent: float = 0.1):
    """Thống kê các lớp ngữ nghĩa xuất hiện có diện tích đạt ngưỡng yêu cầu."""
    flat = mask.reshape(-1)
    total = int(flat.size)
    counts = np.bincount(flat, minlength=NUM_CLASSES).astype(np.int64)

    rows = []
    for class_id in range(1, NUM_CLASSES):
        pixels = int(counts[class_id])
        percent = (pixels / total) * 100.0
        if percent < min_area_percent:
            continue
        rows.append(
            {
                "class_id": class_id,
                "class_name": VOC_CLASSES[class_id],
                "pixels": pixels,
                "percent": percent,
            }
        )
    rows.sort(key=lambda r: r["pixels"], reverse=True)
    return rows


@st.cache_resource(show_spinner=False)
def load_model_safe(checkpoint_path_str: str, device_str: str):
    device = torch.device(device_str)
    try:
        model, ckpt = load_checkpoint_model(checkpoint_path_str, device)
        encoder = ckpt.get("encoder", "resnet50")
        architecture = ckpt.get("architecture", "deeplabv3plus")
        image_size = int(ckpt.get("image_size", 320))
        return model, encoder, architecture, image_size
    except FileNotFoundError as e:
        st.error(f"Không tìm thấy checkpoint: {e}")
        st.stop()
    except RuntimeError as e:
        st.error(f"Checkpoint không tương thích hoặc hỏng: {e}")
        st.stop()


def main():
    st.set_page_config(page_title="DeepLabV3+ Semantic Segmentation UI", layout="wide")
    st.title("DeepLabV3+ - Semantic Segmentation Platform (Pascal VOC 2012)")

    default_data_root = str(VOC_ROOT)
    default_ckpt = str(CHECKPOINT_PATH)

    with st.sidebar:
        st.header("Cấu hình")
        data_root_str = st.text_input("Data root", value=default_data_root)
        ckpt_path_str = st.text_input("Checkpoint", value=default_ckpt)
        split_type = st.selectbox("Loại split", ["benchmark", "smoke"], index=0)
        split = st.selectbox("Split để dự đoán", ["val", "test", "train"], index=0)

        num_samples = st.slider("Số ảnh hiển thị", min_value=1, max_value=12, value=6, step=1)
        random_seed = st.number_input("Seed chọn ảnh", min_value=0, max_value=10_000, value=42, step=1)
        min_area = st.number_input("Diện tích lớp tối thiểu (%)", min_value=0.0, max_value=10.0, value=0.1, step=0.1)

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        st.caption(f"Thiết bị: {device_str}")

    tabs = st.tabs(["Tải ảnh thực tế", "Dự đoán + Trực quan hóa (VOC)", "Đồ thị huấn luyện"])

    data_root = Path(data_root_str)
    ckpt_path = Path(ckpt_path_str)

    with tabs[0]:
        st.subheader("Tải ảnh thực tế lên và phân đoạn các lớp ngữ nghĩa (VOC 20 lớp)")

        uploaded = st.file_uploader("Chọn ảnh (JPG/PNG)", type=["jpg", "jpeg", "png"])
        alpha = st.slider("Độ trong suốt overlay", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        run_btn = st.button("Phân đoạn ảnh đã tải", type="primary")

        if uploaded is None:
            st.info("Hãy tải 1 ảnh lên để bắt đầu.")
        elif run_btn:
            model, encoder, architecture, ckpt_image_size = load_model_safe(str(ckpt_path), device_str)
            try:
                with Image.open(uploaded) as source:
                    image = source.convert("RGB")
            except (OSError, ValueError):
                st.error("Ảnh tải lên không hợp lệ hoặc bị hỏng.")
                st.stop()

            if image.width * image.height > MAX_PIXELS:
                st.error(f"Ảnh quá lớn ({image.width}x{image.height}). Vui lòng tải ảnh dưới {MAX_PIXELS // 1_000_000} Megapixels.")
                st.stop()

            try:
                outputs = predict_with_uncertainty(model, image, ckpt_image_size, torch.device(device_str))
                pred_mask = outputs["hard_mask"]
                entropy_map = outputs["entropy_map"]
                max_prob_map = outputs["max_prob_map"]
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                st.error("GPU không đủ bộ nhớ. Hãy dùng ảnh nhỏ hơn.")
                st.stop()

            image_rgb = np.asarray(image)
            pred_vis = mask_to_color_rgb(pred_mask, ignore_index=IGNORE_INDEX)
            pred_overlay = overlay_mask(image_rgb, pred_vis, alpha)

            st.markdown("#### 1. Kết quả phân đoạn ngữ nghĩa")
            c1, c2, c3 = st.columns(3)
            c1.image(image_rgb, caption="Ảnh gốc (Original)", use_container_width=True)
            c2.image(pred_vis, caption="Mặt nạ phân đoạn (Class Mask)", use_container_width=True)
            c3.image(pred_overlay, caption="Ảnh phủ màu (Overlay)", use_container_width=True)

            st.markdown("#### 2. Bản đồ độ bất định & Độ tin cậy (Uncertainty / Reliability Map)")
            st.caption(
                "Lưu ý: Bản đồ độ bất định thể hiện Normalized Entropy tại độ phân giải gốc của ảnh. "
                "Vùng càng sáng thể hiện ranh giới hoặc đối tượng mà mô hình phân vân nhất, không phải xác suất Bayes đã hiệu chuẩn."
            )
            u1, u2 = st.columns(2)
            u1.image(
                entropy_map,
                caption="Normalized Entropy Map (Vùng sáng = Bất định cao)",
                clamp=True,
                use_container_width=True,
            )
            u2.image(
                max_prob_map,
                caption="Max Softmax Probability Map (Vùng sáng = Độ tin cậy cục bộ cao)",
                clamp=True,
                use_container_width=True,
            )

            st.markdown("### Kết quả lớp ngữ nghĩa xuất hiện (20 lớp VOC)")
            rows = summarize_present_classes(pred_mask, float(min_area))
            if not rows:
                st.warning("Không phát hiện lớp đối tượng nào (ngoài background).")
            else:
                st.dataframe(
                    rows,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "class_id": st.column_config.NumberColumn("Mã lớp", format="%d"),
                        "class_name": st.column_config.TextColumn("Lớp ngữ nghĩa"),
                        "pixels": st.column_config.NumberColumn("Số pixel", format="%d"),
                        "percent": st.column_config.NumberColumn("Tỷ lệ diện tích (%)", format="%.2f"),
                    },
                )

                top_names = ", ".join([r["class_name"] for r in rows[:10]])
                st.success(
                    f"Hoàn tất. Kiến trúc: {architecture.upper()} ({encoder}), "
                    f"Kích thước đầu vào: {ckpt_image_size}. "
                    f"Các lớp chiếm diện tích lớn: {top_names}"
                )

            with st.expander("Danh sách 20 lớp ngữ nghĩa (VOC)"):
                st.write({k: VOC_CLASSES[k] for k in range(1, NUM_CLASSES)})

    with tabs[1]:
        st.subheader("Trực quan hóa Input / Nhãn thật (Ground Truth) / Dự đoán (Prediction)")
        load_btn = st.button("Chạy dự đoán", type="primary")

        if load_btn:
            model, encoder, architecture, ckpt_image_size = load_model_safe(str(ckpt_path), device_str)

            try:
                ids = read_split_ids(data_root, split, split_type=split_type)
            except (FileNotFoundError, ValueError) as e:
                st.error(f"Lỗi đọc tập dữ liệu {split} ({split_type}): {e}")
                st.stop()

            random.seed(int(random_seed))
            chosen = random.sample(ids, k=min(int(num_samples), len(ids)))

            progress = st.progress(0)
            for i, image_id in enumerate(chosen):
                img_path = data_root / "JPEGImages" / f"{image_id}.jpg"
                mask_path = data_root / "SegmentationClass" / f"{image_id}.png"

                try:
                    with Image.open(img_path) as source:
                        image = source.convert("RGB")
                    with Image.open(mask_path) as source:
                        gt_mask = np.asarray(source, dtype=np.int64)
                    gt_vis = mask_to_color_rgb(gt_mask, ignore_index=IGNORE_INDEX)

                    pred_mask = predict_original_size(model, image, ckpt_image_size, torch.device(device_str))
                    pred_vis = mask_to_color_rgb(pred_mask, ignore_index=IGNORE_INDEX)
                    image_rgb = np.asarray(image)

                    c1, c2, c3 = st.columns(3)
                    c1.image(image_rgb, caption=f"Ảnh gốc ({image_id})", use_container_width=True)
                    c2.image(gt_vis, caption="Nhãn thật", use_container_width=True)
                    c3.image(pred_vis, caption="Mặt nạ dự đoán", use_container_width=True)
                except Exception as ex:
                    st.warning(f"Không thể xử lý ảnh {image_id}: {ex}")

                st.write("---")
                progress.progress((i + 1) / len(chosen))

            st.success(f"Hoàn tất. Kiến trúc: {architecture.upper()} ({encoder})")

    with tabs[2]:
        st.subheader("Đồ thị Loss & mIoU theo epoch")
        log_path = data_root.parent / "outputs" / "train_log.csv"

        if not log_path.exists():
            log_path = Path("outputs") / "train_log.csv"

        if not log_path.exists():
            st.warning("Không tìm thấy `outputs/train_log.csv`. Hãy train xong rồi mở tab này.")
        else:
            figure = create_training_figure(log_path)
            st.pyplot(figure)
            figure.clear()
            st.caption(f"Nguồn: {log_path}")


if __name__ == "__main__":
    main()
