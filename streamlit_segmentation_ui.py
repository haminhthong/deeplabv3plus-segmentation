
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from config import IGNORE_INDEX, NUM_CLASSES, VOC_ROOT
from plot_training_curves import read_training_log
from voc_meta import VOC_CLASS_NAMES, mask_to_color_rgb


def overlay_mask(image_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    img = image_rgb.astype(np.float32)
    m = mask_rgb.astype(np.float32)
    out = img * (1.0 - alpha) + m * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def summarize_present_classes(mask: np.ndarray):
    """
    mask: (H, W) int, values 0..20
    trả về danh sách các từ điển: class_id, class_name, pixels, percent
    """
    flat = mask.reshape(-1)
    total = int(flat.size)
    counts = np.bincount(flat, minlength=NUM_CLASSES).astype(np.int64)

    rows = []
    for class_id in range(1, NUM_CLASSES):  # bỏ qua nhãn nền (background)
        pixels = int(counts[class_id])
        if pixels <= 0:
            continue
        rows.append(
            {
                "class_id": class_id,
                "class_name": VOC_CLASS_NAMES.get(class_id, str(class_id)),
                "pixels": pixels,
                "percent": (pixels / total) * 100.0,
            }
        )
    rows.sort(key=lambda r: r["pixels"], reverse=True)
    return rows


def _get_ids(data_root: Path, split: str):
    split_file = data_root / "ImageSets" / "Segmentation" / f"{split}.txt"
    lines = split_file.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path_str: str, device_str: str):
    device = torch.device(device_str)
    ckpt_path = Path(checkpoint_path_str)
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder = ckpt.get("encoder", "resnet50")

    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=None,
        classes=NUM_CLASSES,
        activation=None,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    image_size = int(ckpt.get("image_size", 320))
    return model, encoder, image_size


@st.cache_resource(show_spinner=False)
def build_transforms(image_size: int):
    image_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    mask_resize = transforms.Resize((image_size, image_size), interpolation=Image.NEAREST)
    return image_transform, mask_resize


def main():
    st.set_page_config(page_title="DeepLabV3+ Segmentation UI", layout="wide")
    st.title("DeepLabV3+ - Demo Phân Đoạn Ảnh (Pascal VOC 2012)")

    default_data_root = str(VOC_ROOT)
    default_ckpt = str(Path("outputs") / "deeplabv3plus_voc_best.pth")

    with st.sidebar:
        st.header("Cấu hình")
        data_root_str = st.text_input("Data root", value=default_data_root)
        ckpt_path_str = st.text_input("Checkpoint", value=default_ckpt)
        split = st.selectbox("Split để dự đoán", ["val", "train"], index=0)

        image_size_ui = st.number_input("Image size", min_value=128, max_value=1024, value=320, step=32)
        num_samples = st.slider("Số ảnh hiển thị", min_value=1, max_value=12, value=6, step=1)
        random_seed = st.number_input("Seed chọn ảnh", min_value=0, max_value=10_000, value=42, step=1)

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        st.caption(f"Thiết bị: {device_str}")

    tabs = st.tabs(["Tải ảnh thực tế", "Dự đoán + Trực quan hóa (VOC)", "Đồ thị huấn luyện"])

    data_root = Path(data_root_str)
    ckpt_path = Path(ckpt_path_str)

    if not data_root.exists():
        st.error(f"Không tìm thấy `data_root`: {data_root}")
        return
    if not ckpt_path.exists():
        st.error(f"Không tìm thấy `checkpoint`: {ckpt_path}")
        return

    with tabs[0]:
        st.subheader("Tải ảnh thực tế lên và liệt kê 20 đối tượng (VOC)")

        uploaded = st.file_uploader("Chọn ảnh (jpg/png)", type=["jpg", "jpeg", "png"])
        alpha = st.slider("Độ trong suốt overlay", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        run_btn = st.button("Phân đoạn ảnh đã tải", type="primary")

        if uploaded is None:
            st.info("Hãy tải 1 ảnh lên để bắt đầu.")
        elif run_btn:
            model, encoder, ckpt_image_size = load_model(str(ckpt_path), device_str)
            image_transform, _ = build_transforms(int(image_size_ui))

            image = Image.open(uploaded).convert("RGB")
            input_tensor = image_transform(image).unsqueeze(0).to(next(model.parameters()).device)

            with torch.no_grad():
                pred_logits = model(input_tensor)
                pred_mask = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

            image_resized = np.array(image.resize((int(image_size_ui), int(image_size_ui))))
            pred_vis = mask_to_color_rgb(pred_mask, ignore_index=IGNORE_INDEX)
            pred_overlay = overlay_mask(image_resized, pred_vis, alpha=alpha)

            c1, c2, c3 = st.columns(3)
            c1.image(image_resized, caption="Input", use_container_width=True)
            c2.image(pred_vis, caption="Prediction (Mask)", use_container_width=True)
            c3.image(pred_overlay, caption="Overlay", use_container_width=True)

            st.markdown("### Kết quả liệt kê đối tượng (20 lớp VOC)")
            rows = summarize_present_classes(pred_mask)
            if not rows:
                st.warning("Không phát hiện lớp đối tượng nào (ngoài background).")
            else:
                st.dataframe(
                    rows,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "class_id": st.column_config.NumberColumn("Class ID", format="%d"),
                        "class_name": st.column_config.TextColumn("Object"),
                        "pixels": st.column_config.NumberColumn("Pixels", format="%d"),
                        "percent": st.column_config.NumberColumn("Area (%)", format="%.2f"),
                    },
                )

                top_names = ", ".join([r["class_name"] for r in rows[:10]])
                st.success(
                    f"Hoàn tất. Encoder: {encoder} (checkpoint image_size={ckpt_image_size}). "
                    f"Top objects: {top_names}"
                )

            with st.expander("Danh sách 20 lớp (VOC)"):
                # chỉ hiển thị các lớp đối tượng từ 1 đến 20
                st.write({k: VOC_CLASS_NAMES[k] for k in range(1, NUM_CLASSES)})

    with tabs[1]:
        st.subheader("Trực quan hóa Input / Ground Truth / Prediction")
        load_btn = st.button("Chạy dự đoán", type="primary")

        if load_btn:
            model, encoder, ckpt_image_size = load_model(str(ckpt_path), device_str)

            # Nếu image_size user khác image_size trong checkpoint thì ta vẫn resize theo image_size UI.
            # DeepLabV3+ dự đoán fully-conv nên thường chạy được linh hoạt.
            image_transform, mask_resize = build_transforms(int(image_size_ui))

            ids = _get_ids(data_root, split)
            random.seed(int(random_seed))
            chosen = random.sample(ids, k=min(int(num_samples), len(ids)))

            progress = st.progress(0)
            for i, image_id in enumerate(chosen):
                img_path = data_root / "JPEGImages" / f"{image_id}.jpg"
                mask_path = data_root / "SegmentationClass" / f"{image_id}.png"

                image = Image.open(img_path).convert("RGB")
                gt_mask_img = Image.open(mask_path)

                # Ground truth resize (nearest) và convert numpy int64
                gt_mask = np.array(mask_resize(gt_mask_img), dtype=np.int64)
                gt_vis = mask_to_color_rgb(gt_mask, ignore_index=IGNORE_INDEX)

                input_tensor = image_transform(image).unsqueeze(0)
                input_tensor = input_tensor.to(next(model.parameters()).device)

                with torch.no_grad():
                    pred_logits = model(input_tensor)
                    pred_mask = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

                pred_vis = mask_to_color_rgb(pred_mask, ignore_index=IGNORE_INDEX)
                image_resized = np.array(image.resize((int(image_size_ui), int(image_size_ui))))

                c1, c2, c3 = st.columns(3)
                c1.image(image_resized, caption=f"Input ({image_id})", use_container_width=True)
                c2.image(gt_vis, caption="Ground Truth", use_container_width=True)
                c3.image(pred_vis, caption="Prediction", use_container_width=True)

                st.write("---")
                progress.progress((i + 1) / len(chosen))

            st.success(f"Hoàn tất. Encoder: {encoder} (checkpoint image_size={ckpt_image_size})")

    with tabs[2]:
        st.subheader("Đồ thị Loss & mIoU theo epoch")
        log_path = data_root.parent / "outputs" / "train_log.csv"

        # Cho phép cả trường hợp bạn đặt checkpoint trong outputs/... nhưng vẫn dùng data_root đúng.
        if not log_path.exists():
            # fallback theo đúng cấu trúc hiện tại
            log_path = Path("outputs") / "train_log.csv"

        if not log_path.exists():
            st.warning("Không tìm thấy `outputs/train_log.csv`. Hãy train xong rồi mở tab này.")
        else:
            epochs, train_loss, train_miou, val_loss, val_miou = read_training_log(log_path)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
            axes[0].plot(epochs, val_loss, marker="o", label="Val Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Loss Curve")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(epochs, train_miou, marker="o", label="Train mIoU")
            axes[1].plot(epochs, val_miou, marker="o", label="Val mIoU")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("mIoU")
            axes[1].set_title("mIoU Curve")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            st.pyplot(fig)
            st.caption(f"Nguồn: {log_path}")


if __name__ == "__main__":
    main()

