# DeepLabV3+ Semantic Segmentation trên Pascal VOC 2012

![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)
![CI](https://img.shields.io/badge/CI-Passing-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

Repository triển khai pipeline phân đoạn ngữ nghĩa (Semantic Segmentation) end-to-end bằng **DeepLabV3+**, **PyTorch** và dữ liệu **Pascal VOC 2012**. Dự án được thiết kế chuẩn mực theo 4 tầng phát triển phần mềm AI (Problem, AI/ML Correctness, Software Engineering, và Production/Portfolio).

---

## 1. Định nghĩa bài toán & Phạm vi (Problem Layer)

### 1.1 Phân biệt Semantic Segmentation
Mô hình thực hiện phân đoạn ngữ nghĩa (Semantic Segmentation):
- **Gán nhãn theo Pixel**: Biết chính xác pixel thuộc về lớp ngữ nghĩa nào.
- **Không phân biệt Instance**: Không phân biệt giữa hai đối tượng riêng lẻ cùng một lớp (ví dụ 2 người đứng cạnh nhau).
- **Không đếm số lượng & Không vẽ Bounding Box**: Không tạo bounding box hay đếm số đối tượng.
- **Tỷ lệ diện tích**: Tỷ lệ % đại diện cho phần trăm diện tích pixel thuộc về lớp đó trên tổng số pixel của ảnh, không phải là "độ tin cậy" (confidence score).

### 1.2 Input và Output quy chuẩn
- **Input**:
  - Ảnh RGB định dạng JPG/PNG.
  - Kích thước bất kỳ (tự động xử lý letterbox giữ nguyên tỷ lệ).
  - Nội dung nằm trong phạm vi 20 lớp đối tượng Pascal VOC + 1 lớp Nền (Background).
- **Output**:
  - Mask nhãn Class ID 2D (cùng kích thước với ảnh gốc).
  - Mask màu trực quan theo chuẩn màu Pascal VOC.
  - Ảnh phủ màu (Overlay blend).
  - Bảng thống kê tỷ lệ diện tích pixel (%) của từng lớp ngữ nghĩa xuất hiện.

---

## 2. Thử nghiệm AI/ML & Đánh giá (AI/ML Correctness)

### 2.1 Cấu trúc Split dữ liệu cố định
Dữ liệu được quản lý qua thư mục `splits/` cố định để chống rò rỉ (leakage):
- `splits/train.txt`: Dùng để huấn luyện mô hình.
- `splits/val.txt`: Dùng để chọn checkpoint, tinh chỉnh hyperparameter và early stopping.
- `splits/test.txt`: Tập kiểm thử độc lập cuối cùng, chỉ được đánh giá 1 lần duy nhất sau khi đã khóa cấu hình mô hình.

### 2.2 Kiểm tra Rò rỉ Dữ liệu & Integrity Audit
Sử dụng script audit dữ liệu `validate_dataset.py` dựa trên SHA-256 hash ảnh:
```bash
python validate_dataset.py --data-root data/VOC2012_train_val/VOC2012_train_val --splits-dir splits
```
Output báo cáo JSON tự động kiểm tra:
- SHA-256 hash trùng lặp trong cùng 1 split hoặc giữa train/val/test.
- Đầy đủ cặp ảnh JPG và mask PNG.
- Khớp kích thước giữa ảnh và mask.
- Nhãn mask nằm trong dải hợp lệ `[0..20]` hoặc `255`.

### 2.3 Bảng so sánh Baseline vs Champion

| Mô hình | Architecture | Backbone | Params | mIoU (All) | mIoU (No BG) | Mean Dice | Pixel Acc | Latency (CPU/GPU) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | U-Net / FCN | ResNet50 | Đo thật | Đo thật | Đo thật | Đo thật | Đo thật | Đo thật |
| **Champion** | **DeepLabV3+** | **ResNet50** | **Đo thật** | **Đo thật** | **Đo thật** | **Đo thật** | **Đo thật** | **Đo thật** |

---

## 3. Cài đặt & Sử dụng (Software Engineering)

### 3.1 Cài đặt môi trường
```bash
# Tạo môi trường ảo Python
python -m venv .venv
source .venv/bin/activate  # Hoặc .venv\Scripts\Activate.ps1 trên Windows

# Cài đặt thư viện phát triển và kiểm thử
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3.2 Chạy bộ kiểm thử (Unit Tests) & CI
Repository tích hợp bộ test tự động (`pytest`) và linter (`ruff`):
```bash
# Chạy kiểm tra style code
ruff check .

# Chạy biên dịch kiểm tra syntax
python -m compileall -q .

# Chạy bộ test unit đầy đủ
pytest -q
```

### 3.3 Huấn luyện mô hình (Training)
```bash
python train_deeplabv3plus.py \
  --architecture deeplabv3plus \
  --encoder resnet50 \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --seed 42 \
  --deterministic
```

### 3.4 Đánh giá Checkpoint (Evaluation)
Đánh giá trên tập `test` cố định và xuất báo cáo JSON & CSV chi tiết từng lớp:
```bash
python evaluate.py \
  --checkpoint outputs/deeplabv3plus_resnet50_voc_best.pth \
  --split test \
  --output outputs/evaluation.json \
  --csv-output outputs/per_class_metrics.csv
```

### 3.5 Tải Checkpoint công khai
Tải checkpoint đã được xác minh bằng checksum SHA-256:
```bash
python download_checkpoint.py --output outputs/deeplabv3plus_voc_best.pth
```

### 3.6 Giao diện Demo Streamlit
```bash
streamlit run streamlit_segmentation_ui.py
```

---

## 4. Định hướng Production & Tối ưu (Production Layer)

### 4.1 Kiến trúc Microservice khuyến nghị
Để phục vụ số lượng lớn người dùng (100+ concurrent users), không nên sử dụng trực tiếp Streamlit làm backend suy luận. Kiến trúc chuẩn:
```text
Browser Client / Frontend (Streamlit / React)
       ↓ (HTTP REST / Base64 / Multipart)
FastAPI Backend Server
       ↓ (Request Queue / Redis)
Inference Worker Process (Torch / ONNX Runtime)
       ↓
GPU / CUDA Accelerator
```

### 4.2 API Endpoint tiêu chuẩn (FastAPI)
- `GET /health`: Kiểm tra trạng thái dịch vụ và GPU.
- `GET /model-info`: Trả về thông tin mô hình, phiên bản, encoder và kích thước đầu vào.
- `POST /predict`: Nhận file ảnh upload, trả về mask PNG / Base64, danh sách lớp ngữ nghĩa xuất hiện và độ trễ (`latency_ms`).

### 4.3 Giới hạn tài nguyên & Bảo mật
- **Kích thước ảnh**: Tối đa 10 MB per upload, giới hạn 20 Megapixels (`MAX_PIXELS = 20_000_000`).
- **Quyền riêng tư (Privacy Notice)**: Dịch vụ không lưu trữ dữ liệu ảnh của người dùng trên ổ đĩa lâu dài, xóa toàn bộ artifact tạm sau request và không sử dụng ảnh upload để huấn luyện lại mô hình khi chưa được phép.

---

## Cấu trúc Repository

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml               # Workflow GitHub Actions CI
├── splits/
│   ├── train.txt                # Split train cố định
│   ├── val.txt                  # Split validation cố định
│   └── test.txt                 # Split test cuối cố định
├── tests/
│   ├── test_dataset.py          # Test kiểm tra dataset & leakage
│   ├── test_metrics.py          # Test tính toán mIoU/Dice
│   ├── test_inference.py        # Test suy luận với mock model
│   └── test_checkpoint.py       # Test lưu/đọc checkpoint & metadata
├── config.py                    # Cấu hình hằng số hệ thống
├── dataset_voc.py               # Dataset class & letterbox transform dùng chung
├── download_checkpoint.py       # Script tải checkpoint & kiểm tra SHA-256
├── evaluate.py                  # Script đánh giá metric & latency
├── inference.py                 # Hàm dựng model, letterbox & predict
├── metrics.py                   # Class tính toán IoU, Dice, Pixel Acc & export CSV
├── plot_training_curves.py      # Vẽ biểu đồ loss và mIoU
├── requirements.txt             # Thư viện runtime
├── requirements-dev.txt         # Thư viện kiểm thử (pytest, ruff)
├── streamlit_segmentation_ui.py # Giao diện Streamlit UI demo
├── train_deeplabv3plus.py       # Pipeline huấn luyện mô hình mô-đun
├── validate_dataset.py          # Script audit dataset & SHA-256 leakage
├── visualize_predictions.py     # Xuất ảnh trực quan hóa 4 cột
└── voc_meta.py                  # Tên 20 lớp VOC và colormap RGB
```

---

## License

Dự án được phát hành theo giấy phép MIT. Xem [LICENSE](LICENSE) để biết chi tiết.
