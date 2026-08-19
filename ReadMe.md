# DeepLabV3+ Semantic Segmentation

[![CI](https://github.com/OWNER/REPOSITORY/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPOSITORY/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Pipeline phân đoạn ngữ nghĩa Pascal VOC 2012 dùng DeepLabV3+ và PyTorch. Repository gồm huấn luyện, metric, resume checkpoint, trực quan hóa dự đoán và demo Streamlit cho ảnh tải lên.

> Thay `OWNER/REPOSITORY` trong badge CI sau khi đưa repository lên GitHub.

## Kết quả

Không công bố số liệu chưa được kiểm chứng. Sau khi huấn luyện, metric tốt nhất được ghi vào `outputs/best_metrics.json` và lịch sử từng epoch vào `outputs/train_log.csv`.

| Model | Backbone | Input | Loss | Val mIoU | Mean Dice | Pixel Accuracy |
|---|---|---:|---|---:|---:|---:|
| DeepLabV3+ | ResNet50 | 320 | CE + 0.5 Dice | Chưa đo | Chưa đo | Chưa đo |

Khi có kết quả đã xác nhận, cập nhật bảng cùng cấu hình GPU, thời gian huấn luyện, ảnh dự đoán tốt và failure cases. Checkpoint lớn nên được phát hành qua GitHub Release hoặc Hugging Face; repository hiện không tự nhận là có checkpoint công khai.

## Điểm chính

- Nhãn VOC: `0` là background, `1–20` là đối tượng, `255` là vùng void bị bỏ qua.
- Train bằng random scale, crop/pad, horizontal flip, affine nhẹ và color jitter.
- Validation/inference giữ tỷ lệ ảnh và pad thay vì kéo ảnh thành hình vuông.
- Metric gồm mIoU, mIoU không background, per-class IoU, mean Dice, pixel accuracy và mean class accuracy.
- Checkpoint chứa model, optimizer, scheduler, AMP scaler, epoch, tham số train và Git commit.
- Demo upload chỉ cần checkpoint; dataset chỉ cần cho tab đánh giá VOC.
- Mask dự đoán được đưa về đúng kích thước ảnh gốc trước khi overlay.

## Cài đặt

Yêu cầu Python 3.11. Với CUDA, nên cài bản PyTorch phù hợp trước, rồi cài dependency còn lại.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Dataset mặc định:

```text
data/VOC2012_train_val/VOC2012_train_val/
├── JPEGImages/
├── SegmentationClass/
└── ImageSets/Segmentation/
    ├── train.txt
    └── val.txt
```

Có thể chỉ định vị trí khác bằng `--data-root`.

## Huấn luyện

```bash
python train_deeplabv3plus.py --epochs 50 --batch-size 8 \
  --image-size 320 --amp --patience 8
```

Tiếp tục một lần chạy:

```bash
python train_deeplabv3plus.py --epochs 80 \
  --resume outputs/deeplabv3plus_voc_best.pth
```

Artifact trong `outputs/`:

- `deeplabv3plus_voc_best.pth`: checkpoint có validation mIoU tốt nhất.
- `best_metrics.json`: metric chi tiết của checkpoint tốt nhất.
- `train_log.csv`: loss và mIoU theo epoch.

## Dự đoán và trực quan hóa

Checkpoint đã huấn luyện là bắt buộc vì decoder segmentation không có ý nghĩa nếu chỉ dùng encoder ImageNet.

```bash
python visualize_predictions.py \
  --checkpoint outputs/deeplabv3plus_voc_best.pth --indices 0 1 2 3 4
```

Ảnh Image / Ground truth / Prediction / Overlay được lưu vào `outputs/viz/`. Vẽ learning curves bằng `python plot_training_curves.py`.

## Demo Streamlit

```bash
python -m streamlit run streamlit_segmentation_ui.py
```

Tab **Tải ảnh thực tế** hoạt động khi có checkpoint, không yêu cầu Pascal VOC. Tab đánh giá VOC kiểm tra dataset riêng khi bấm chạy. Thanh bên có ngưỡng diện tích tối thiểu để lọc lớp xuất hiện do vài pixel nhiễu.

## Kiểm thử

```bash
python -m compileall -q .
pytest -q
```

GitHub Actions chạy hai lệnh cho mỗi push và pull request.

## Cấu trúc

```text
├── config.py
├── dataset_voc.py
├── inference.py
├── metrics.py
├── train_deeplabv3plus.py
├── visualize_predictions.py
├── streamlit_segmentation_ui.py
├── plot_training_curves.py
├── voc_meta.py
├── tests/
├── .github/workflows/ci.yml
└── requirements.txt
```

## Hạn chế và roadmap

- Chưa có checkpoint công khai hoặc demo online.
- Chưa có ablation xác nhận cho loss, backbone và image size.
- Chưa benchmark latency/FPS, peak VRAM hoặc xuất ONNX.
- Pipeline hiện tối ưu cho Pascal VOC 2012 và 21 lớp cố định.

Thí nghiệm tiếp theo nên so sánh CE với CE + Dice, backbone nặng/nhẹ, input 320/512 và DeepLabV3+ với U-Net/FCN baseline.

## Tài liệu tham khảo

- Chen et al., *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*.
- Everingham et al., *The Pascal Visual Object Classes Challenge*.
- `segmentation-models-pytorch`.

## License

MIT — xem [LICENSE](LICENSE).
