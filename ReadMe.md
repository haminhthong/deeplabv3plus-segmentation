# DeepLabV3+ Semantic Segmentation Platform

> **Leakage-Audited Training, Multi-Architecture Benchmarking & Original-Resolution Inference on Pascal VOC**

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CI](https://github.com/haminhthong/deeplabv3plus-segmentation/actions/workflows/ci.yml/badge.svg)](https://github.com/haminhthong/deeplabv3plus-segmentation/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Repository cung cấp một nền tảng phân đoạn ngữ nghĩa (Semantic Segmentation) chuyên sâu, toàn diện và có thể tái lập hoàn toàn trên tập dữ liệu **Pascal VOC 2012** (20 lớp đối tượng + 1 lớp nền). Dự án triển khai quy trình thẩm định dữ liệu chống rò rỉ (anti-leakage audit), so sánh đa kiến trúc encoder-decoder (U-Net, FPN, DeepLabV3+), đánh giá chi tiết theo đường biên (Boundary F1), phân khúc kích thước vùng (region size slices), bản đồ độ bất định (uncertainty map) và suy luận letterbox tại độ phân giải gốc của ảnh.

---

## Kiến Trúc Hệ Thống Chuẩn Hóa (System Architecture)

Dự án được chuẩn hóa theo 2 pipeline canonical thống nhất xuyên suốt toàn bộ mã nguồn:

### 1. Offline ML Pipeline
```text
                    OFFLINE ML PIPELINE

Pascal VOC Images + Masks
          ↓
1. DATA AUDIT
   ├── image/mask pairing
   ├── dimension consistency
   ├── valid class IDs
   ├── duplicate IDs
   └── SHA-256 exact duplicates
          ↓
2. DATA SPLIT CONTRACT
   ├── Development Train (~70%)
   ├── Validation (~15%)
   └── Locked Custom Holdout (~15%)
          ↓
3. TRAIN TRANSFORM
   Image + Mask
   ├── random scale [0.75, 1.5]
   ├── unbiased padding
   ├── joint crop
   ├── horizontal flip
   ├── affine
   └── mild color jitter (image only)
          ↓
4. MODEL CANDIDATES
   ├── Candidate A: U-Net
   ├── Candidate B: FPN
   └── Candidate C: DeepLabV3+
          ↓
5. TRAINING
   ImageNet Encoder (ResNet50)
        ↓
   Segmentation Decoder
        ↓
   Loss: Cross-Entropy + 0.5 * Dice
        ↓
   AdamW + Cosine Annealing Schedule
          ↓
6. VALIDATION MODEL SELECTION
   ├── mIoU (All classes)
   ├── mIoU (No background)
   ├── Mean Dice
   ├── per-class IoU
   └── latency / model size
          ↓
     Champion Selection
          ↓
7. LOCK MODEL + POLICY
          ↓
8. FINAL HOLDOUT EVALUATION
   ├── pixel metrics (mIoU, Acc)
   ├── class metrics (per-class IoU/Dice)
   ├── boundary metrics (Boundary F1)
   ├── region-size slices (small / med / large)
   └── error analysis (confusion pairs)
          ↓
9. VERSIONED ARTIFACT
   checkpoint + config + split hashes
   + git SHA + metrics + model card
```

### 2. Online Serving Pipeline
```text
                    ONLINE PIPELINE

RGB Image (Any Resolution)
   ↓
Input Validation (Format, Max Pixels <= 20M)
   ↓
Letterbox Preprocessing + Normalization (Aspect-ratio preserved)
   ↓
Segmentation Model Forward Pass (GPU / CPU)
   ↓
21-Class Logits Map
   ↓
Remove Padding (Letterbox region slice)
   ↓
Bilinear Resize Logits → Original Resolution (H × W)
   ↓
Softmax Probability Distribution
   ↓
Argmax Mask & Reliability Maps
   ├── Class-ID 2D Mask
   ├── Pascal Color Visualization Mask
   ├── Alpha Blended Overlay
   ├── Semantic Area Coverage (%)
   └── Uncertainty / Reliability Map (Normalized Entropy & Max Prob)
```

---

## 1. Định Nghĩa Bài Toán & Phạm Vi (Problem Layer)

### 1.1 Nguyên lý Semantic Segmentation
- **Gán nhãn theo Pixel**: Dự đoán chính xác mỗi pixel thuộc về lớp nào trong 21 lớp VOC.
- **Không phân biệt Instance**: Không tách biệt các đối tượng riêng lẻ cùng lớp (không sinh bounding box).
- **Tỷ lệ diện tích (%)**: Tỷ lệ phần trăm diện tích pixel thuộc về lớp đối tượng trên tổng số pixel của ảnh, không phải là xác suất tự tin (confidence score).
- **Bản đồ độ bất định (Uncertainty / Reliability Map)**: Thể hiện độ phân vân (Normalized Entropy) của phân bố Softmax tại độ phân giải gốc của ảnh, giúp kiểm toán các vùng ranh giới phức tạp, không phải là xác suất Bayes đã hiệu chuẩn.

### 1.2 Quy ước Tiền Xử Lý (Transform Contract)
> [!NOTE]
> - **Training Transform**: Sử dụng scale ngẫu nhiên $[0.75, 1.5]$, đệm ngẫu nhiên không thiên lệch góc (unbiased padding), crop ngẫu nhiên về target $320 \times 320$, lật ngang, affine và color jitter nhẹ (chỉ trên ảnh RGB).
> - **Validation & Serving Transform**: Sử dụng deterministic letterbox giữ nguyên tỷ lệ khung hình với đệm đều vào tâm, bảo toàn hình thái học thực tế của vật thể.

---

## 2. Thử Nghiệm AI/ML & Đánh Giá Đa Chiều (AI/ML Layer)

### 2.1 Hợp Đồng Phân Chia Dữ Liệu (Split Contract)
> [!WARNING]
> **Smoke split (`splits/smoke/`) chỉ dùng cho kiểm thử phần mềm / CI smoke testing và tuyệt đối không được sử dụng để báo cáo hiệu năng mô hình.**

Dữ liệu được quản lý tường minh thành 2 bộ split:
- **`splits/smoke/`** (5 train, 4 val, 4 test): Phục vụ chạy kiểm thử nhanh (quick smoke test) và CI.
- **`splits/benchmark/`** (chia 3 tầng có chủ đích, quản lý qua `split_manifest.json`):
  - `train.txt`: Development Train (~70%) dùng để tối ưu trọng số.
  - `val.txt`: Validation Set (~15%) dùng để chọn checkpoint, tune hyperparameter và early stopping.
  - `test.txt`: Locked Custom Holdout (~15%) được niêm phong và chỉ đánh giá duy nhất một lần khi đã chốt mô hình Champion.

Tự động sinh hoặc kiểm toán split bằng script:
```bash
python scripts/create_benchmark_splits.py --data-root data/VOC2012_train_val/VOC2012_train_val --seed 42
```

### 2.2 Thẩm Định Dữ Liệu & Chống Rò Rỉ (Data Integrity & Anti-Leakage)
Script `validate_dataset.py` thực hiện kiểm toán toàn diện:
```bash
python validate_dataset.py --split-type benchmark --generate-manifest splits/benchmark/split_manifest.json
```
Nội dung kiểm tra tự động:
1. **Trùng lặp ID**: Kiểm tra không có ID lặp trong cùng split (intra-split) hoặc giữa các split (inter-split).
2. **Trùng lặp SHA-256**: Phát hiện các ảnh có nội dung giống hệt nhau nhưng khác ID.
3. **Tính toàn vẹn**: Khớp đầy đủ cặp JPG và PNG, đồng nhất kích thước ảnh và mask.
4. **Phân bố lớp**: Thống kê số ảnh/lớp, số pixel/lớp, và tỷ lệ tiền cảnh/hậu cảnh (foreground/background ratio).

### 2.3 Bảng So Sánh Ứng Viên (Candidate Benchmark Leaderboard)

Cả 3 ứng viên được huấn luyện dưới cùng một giao thức so sánh công bằng:
- Cùng encoder ImageNet: **ResNet50**
- Cùng kích thước đầu vào: **320 × 320**
- Cùng hàm mất mát: **Cross-Entropy + 0.5 × Dice**
- Cùng optimizer: **AdamW**, Cosine Annealing schedule, batch size 8
- Chỉ khác biệt về kiến trúc decoder:

| Ứng Viên | Architecture | Encoder | Val mIoU (All) | Val mIoU (No BG) | Mean Dice | Boundary F1 | Latency (p50 / p95) | Trạng Thái |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Candidate A** | U-Net | ResNet50 | *Sẵn sàng đo* | *Sẵn sàng đo* | *Sẵn sàng đo* | *Sẵn sàng đo* | *Sẵn sàng đo* | Baseline Candidate |
| **Candidate B** | FPN | ResNet50 | *Sẵn sàng đo* | *Sẵn sàng đo* | *Sẵn sàng đo* | *Sẵn sàng đo* | *Sẵn sàng đo* | Multi-scale Candidate |
| **Candidate C** | DeepLabV3+ | ResNet50 | *Sẵn sàng đo* | *Sẵn sàng đo* | *Sẵn sàng đo* | *Sẵn sàng đo* | *Sẵn sàng đo* | ASPP Candidate |

> [!NOTE]
> Mô hình chỉ được vinh danh là **Champion** sau khi hoàn thành đo kiểm thực tế trên tập **Validation** và đạt điểm mIoU vượt trội. Mô hình Champion sau đó sẽ được khóa và đánh giá một lần duy nhất trên tập **Locked Custom Holdout**.

---

## 3. Cài Đặt & Hướng Dẫn Sử Dụng (Software Engineering)

### 3.1 Cài đặt Môi trường
```bash
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\Activate.ps1

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3.2 Kiểm thử Tự động (Testing Suite) & Linter
```bash
# Kiểm tra code style với Ruff
python -m ruff check .

# Biên dịch kiểm tra cú pháp Python
python -m compileall -q .

# Chạy toàn bộ bộ kiểm thử tự động
python -m pytest -v
```

### 3.3 Huấn Luyện Mô Hình (Training)
```bash
# Huấn luyện Candidate C (DeepLabV3+) trên benchmark split chuẩn
python train_deeplabv3plus.py \
  --architecture deeplabv3plus \
  --encoder resnet50 \
  --split-type benchmark \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-4 \
  --seed 42 \
  --deterministic

# Huấn luyện Candidate A (U-Net) để so sánh baseline
python train_deeplabv3plus.py \
  --architecture unet \
  --encoder resnet50 \
  --split-type benchmark \
  --epochs 50

# Huấn luyện nhanh kiểm tra code trên smoke split
python train_deeplabv3plus.py \
  --architecture fpn \
  --split-type smoke \
  --epochs 2
```

### 3.4 Đánh Giá Checkpoint (Evaluation)
Đánh giá toàn diện bao gồm Boundary F1, region-size slicing, confusion pairs và đo độ trễ chi tiết (p50/p95):
```bash
python evaluate.py \
  --checkpoint outputs/deeplabv3plus_resnet50_voc_best.pth \
  --split test \
  --split-type benchmark \
  --batch-size 8 \
  --output outputs/evaluation.json \
  --csv-output outputs/per_class_metrics.csv
```

### 3.5 Tải Checkpoint Đã Xác Minh
> [!NOTE]
> Checkpoint chính thức v1.0.0 đang trong giai đoạn đo kiểm và đóng gói release. Bạn có thể tự huấn luyện mô hình bằng `train_deeplabv3plus.py` hoặc tải từ URL tùy chọn của bạn:
```bash
python download_checkpoint.py \
  --url https://your-server.com/deeplabv3plus_voc_best.pth \
  --sha256 <EXPECTED_SHA256>
```

### 3.6 Trực Quan Hóa & Demo Giao Diện (Streamlit UI)
Khởi chạy ứng dụng web demo trực quan hóa kết quả phân đoạn và bản đồ bất định (Uncertainty Map):
```bash
streamlit run streamlit_segmentation_ui.py
```

---

## 4. Định Hướng Triển Khai Production (Production Roadmap)

> [!NOTE]
> Repository hiện tại tập trung vào triển khai cục bộ (Local Training, CLI Tools, Streamlit Demo). Sơ đồ dưới đây mô tả kiến trúc vi dịch vụ (Microservices) khuyến nghị cho giai đoạn mở rộng phục vụ production tải cao.

### 4.1 Kiến trúc Vi Dịch Vụ Khuyến Nghị
```text
Client Browser / Mobile App
       ↓ (HTTPS REST API / WebSocket)
FastAPI Gateway Server (Auth, Rate Limiting, Input Validation)
       ↓ (Message Broker / Redis Streams)
Inference Workers Pool (PyTorch C++ LibTorch / ONNX Runtime / TensorRT)
       ↓
GPU Cluster (NVIDIA Triton Inference Server)
```

### 4.2 Giới Hạn Tài Nguyên & An Toàn Dữ Liệu
- **Giới hạn kích thước ảnh**: Giới hạn tối đa 10 MB per file, tối đa 20 Megapixels (`MAX_PIXELS = 20_000_000`) nhằm bảo vệ bộ nhớ GPU khỏi lỗi Out-Of-Memory.
- **Quyền riêng tư (Privacy Notice)**: Hệ thống suy luận phục vụ theo chế độ stateless; toàn bộ hình ảnh tạm thời được giải phóng ngay sau khi phản hồi hoàn tất.

---

## Cấu Trúc Repository

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml                     # Workflow GitHub Actions CI tự động
├── docs/
│   ├── code_audit.md                  # Báo cáo kiểm toán toàn vẹn mã nguồn
│   └── improvement_guide.md           # Hướng dẫn nâng cấp 4 tầng AI software
├── scripts/
│   └── create_benchmark_splits.py     # Script tạo benchmark split và split_manifest.json
├── splits/
│   ├── benchmark/                     # Benchmark split có chủ đích (Train/Val/Holdout)
│   │   ├── split_manifest.json        # Metadata, SHA-256 hashes & phân bố lớp
│   │   ├── train.txt                  # Tập Development Train
│   │   ├── val.txt                    # Tập Validation
│   │   └── test.txt                   # Tập Locked Custom Holdout
│   └── smoke/                         # Smoke split dùng cho kiểm thử nhanh phần mềm
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
├── tests/
│   ├── test_checkpoint.py             # Kiểm thử lưu và đọc checkpoint metadata
│   ├── test_dataset.py                # Kiểm thử dataset, split contracts & leakage
│   ├── test_inference.py              # Kiểm thử build model (unet/fpn/deeplabv3plus) & uncertainty map
│   └── test_metrics.py                # Kiểm thử mIoU, Dice, Boundary F1 & confusion analysis
├── config.py                          # Hằng số cấu hình hệ thống
├── dataset_voc.py                     # Dataset VOC, joint augmentations & letterbox contract
├── download_checkpoint.py             # Script tải checkpoint & kiểm tra SHA-256
├── evaluate.py                        # Đánh giá toàn diện: mIoU, Boundary F1, Latency p50/p95
├── inference.py                       # Dựng model, predict letterbox & uncertainty mapping
├── metrics.py                         # Tính toán mIoU, Boundary F1, region-size slicing, confusion pairs
├── plot_training_curves.py            # Vẽ đồ thị Loss & mIoU
├── requirements-dev.txt               # Thư viện phát triển & kiểm thử (pytest, ruff)
├── requirements.txt                   # Thư viện runtime (torch 2.7.1, torchvision, SMP)
├── streamlit_segmentation_ui.py       # Giao diện web demo tương tác & uncertainty map
├── train_deeplabv3plus.py             # Pipeline huấn luyện mô hình đa kiến trúc
├── validate_dataset.py                # Script audit dữ liệu, chống rò rỉ & phân bố lớp
├── visualize_predictions.py           # Xuất ảnh trực quan hóa 4 cột
└── voc_meta.py                        # Danh sách 20 lớp VOC & bảng màu RGB chuẩn
```

---

## License

Dự án được phát hành theo giấy phép MIT. Xem [LICENSE](LICENSE) để biết thêm thông tin.
