# DeepLabV3+ Semantic Segmentation trên Pascal VOC 2012

[![CI](https://github.com/OWNER/REPOSITORY/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPOSITORY/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Dự án xây dựng pipeline phân đoạn ngữ nghĩa hoàn chỉnh bằng **DeepLabV3+**, **PyTorch** và **Pascal VOC 2012**. Mỗi pixel trong ảnh được gán vào background hoặc một trong 20 lớp đối tượng như người, ô tô, xe đạp, chó và mèo.

Repository hỗ trợ toàn bộ vòng đời thí nghiệm: đọc dữ liệu, augmentation đồng bộ ảnh–mask, huấn luyện, validation, lưu/resume checkpoint, đánh giá nhiều metric, trực quan hóa và demo Streamlit cho ảnh người dùng tải lên.

> Trước khi public repository, thay `OWNER/REPOSITORY` trong badge CI bằng tên GitHub repository thực tế.

## Mục lục

- [Tổng quan kỹ thuật](#tổng-quan-kỹ-thuật)
- [Kết quả thực nghiệm](#kết-quả-thực-nghiệm)
- [Cài đặt](#cài-đặt)
- [Chuẩn bị Pascal VOC](#chuẩn-bị-pascal-voc)
- [Huấn luyện](#huấn-luyện)
- [Resume và checkpoint](#resume-và-checkpoint)
- [Metric đánh giá](#metric-đánh-giá)
- [Trực quan hóa](#trực-quan-hóa)
- [Demo Streamlit](#demo-streamlit)
- [Kiểm thử và CI](#kiểm-thử-và-ci)
- [Cấu trúc repository](#cấu-trúc-repository)
- [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)
- [Hạn chế và roadmap](#hạn-chế-và-roadmap)

## Tổng quan kỹ thuật

### Bài toán

Semantic segmentation dự đoán một class ID cho từng pixel. Khác với object detection, kết quả không chỉ là bounding box mà là vùng hình dạng chính xác của đối tượng.

Pascal VOC dùng quy ước nhãn:

| Giá trị | Ý nghĩa |
|---:|---|
| `0` | Background |
| `1–20` | 20 lớp đối tượng Pascal VOC |
| `255` | Void/ignore region, không dùng khi tính loss và metric |

### Mô hình

- Kiến trúc: DeepLabV3+.
- Backbone mặc định: ResNet50 pretrained trên ImageNet.
- Số lớp đầu ra: 21, bao gồm background.
- Thư viện mô hình: `segmentation-models-pytorch`.
- Optimizer: AdamW, `weight_decay=1e-4`.
- Scheduler: Cosine Annealing.
- Loss: Cross Entropy + `0.5 × Dice Loss`.
- Checkpoint tốt nhất được chọn theo validation mIoU.

DeepLabV3+ kết hợp atrous convolution, ASPP và decoder để thu thập ngữ cảnh đa tỷ lệ trong khi phục hồi biên đối tượng. Backbone có thể thay bằng encoder khác được thư viện hỗ trợ thông qua `--encoder`.

### Pipeline dữ liệu

Pipeline training áp dụng cùng một phép biến đổi hình học cho ảnh và mask:

1. Random scale trong khoảng `0.75–1.5`.
2. Pad nếu ảnh nhỏ hơn kích thước đầu vào.
3. Random crop về kích thước cố định.
4. Random horizontal flip.
5. Random affine nhẹ.
6. Color jitter chỉ trên ảnh RGB.
7. Chuẩn hóa theo mean/std ImageNet.

Mask luôn dùng nearest-neighbor interpolation để không sinh class ID không hợp lệ. Vùng mới tạo bởi padding hoặc affine được gán `255` và bị bỏ qua trong loss/metric.

Validation và inference giữ nguyên tỷ lệ ảnh, resize cạnh phù hợp rồi pad thành hình vuông. Sau inference, phần padding bị loại bỏ và logits được nội suy về đúng kích thước ảnh gốc trước khi tạo mask.

## Kết quả thực nghiệm

Repository không điền số liệu chưa được chạy và xác minh. Sau khi huấn luyện:

- `outputs/train_log.csv` lưu loss và mIoU theo epoch.
- `outputs/best_metrics.json` lưu metric chi tiết của checkpoint tốt nhất.
- `outputs/deeplabv3plus_voc_best.pth` lưu checkpoint tốt nhất.

| Model | Backbone | Input | Loss | Val mIoU | mIoU không BG | Mean Dice | Pixel Accuracy |
|---|---|---:|---|---:|---:|---:|---:|
| DeepLabV3+ | ResNet50 | 320×320 | CE + 0.5 Dice | Chưa đo | Chưa đo | Chưa đo | Chưa đo |

Khi công bố kết quả, nên bổ sung:

- GPU/CPU và dung lượng VRAM.
- Số epoch, batch size, learning rate và seed.
- Thời gian huấn luyện.
- Latency/FPS trên CPU và GPU.
- Biểu đồ per-class IoU.
- Ảnh dự đoán tốt, trường hợp thất bại và phân tích nguyên nhân.

## Cài đặt

### Yêu cầu

- Python 3.11 khuyến nghị.
- Windows, Linux hoặc macOS.
- NVIDIA GPU có CUDA là tùy chọn; CPU vẫn chạy được nhưng training chậm hơn.

### Tạo môi trường

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu cần CUDA, hãy cài PyTorch wheel tương ứng với phiên bản CUDA của máy theo hướng dẫn chính thức của PyTorch trước khi cài các package còn lại.

Kiểm tra môi trường:

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Chuẩn bị Pascal VOC

Đường dẫn mặc định được cấu hình trong `config.py`:

```text
data/VOC2012_train_val/VOC2012_train_val/
├── JPEGImages/
│   ├── 2007_000032.jpg
│   └── ...
├── SegmentationClass/
│   ├── 2007_000032.png
│   └── ...
└── ImageSets/
    └── Segmentation/
        ├── train.txt
        └── val.txt
```

Mỗi dòng trong `train.txt` hoặc `val.txt` chứa ID ảnh không có phần mở rộng. Ví dụ `2007_000032` tương ứng với:

- `JPEGImages/2007_000032.jpg`
- `SegmentationClass/2007_000032.png`

Không bắt buộc đặt dataset tại đường dẫn mặc định. Dùng `--data-root` để chọn vị trí khác.

## Huấn luyện

Chạy cấu hình mặc định:

```bash
python train_deeplabv3plus.py
```

Ví dụ cấu hình đầy đủ:

```bash
python train_deeplabv3plus.py \
  --data-root data/VOC2012_train_val/VOC2012_train_val \
  --epochs 50 \
  --batch-size 8 \
  --lr 0.0001 \
  --image-size 320 \
  --num-workers 2 \
  --encoder resnet50 \
  --encoder-weights imagenet \
  --seed 42 \
  --output-dir outputs \
  --amp \
  --patience 8
```

### Tham số chính

| Tham số | Mặc định | Mô tả |
|---|---:|---|
| `--data-root` | từ `config.py` | Thư mục gốc Pascal VOC |
| `--epochs` | `50` | Tổng số epoch, kể cả khi resume |
| `--batch-size` | `8` | Batch size |
| `--lr` | `1e-4` | Learning rate ban đầu |
| `--image-size` | `320` | Kích thước crop/pad đầu vào |
| `--num-workers` | `2` | Số DataLoader worker |
| `--encoder` | `resnet50` | Backbone của DeepLabV3+ |
| `--encoder-weights` | `imagenet` | Trọng số khởi tạo encoder |
| `--seed` | `42` | Seed cho Python, NumPy và PyTorch |
| `--output-dir` | `outputs` | Nơi lưu artifact |
| `--amp` / `--no-amp` | bật | Mixed precision trên CUDA |
| `--patience` | `0` | Early stopping; `0` là tắt |
| `--resume` | không có | Checkpoint để tiếp tục training |

Nếu hết VRAM, giảm `--batch-size` trước, sau đó giảm `--image-size`. AMP chỉ được kích hoạt thực tế khi chạy CUDA.

## Resume và checkpoint

Tiếp tục từ checkpoint:

```bash
python train_deeplabv3plus.py \
  --resume outputs/deeplabv3plus_voc_best.pth \
  --epochs 80
```

`--epochs 80` nghĩa là dừng ở epoch 80, không phải train thêm 80 epoch. Checkpoint lưu:

- Epoch hiện tại.
- Model state.
- Optimizer state.
- Scheduler state.
- AMP scaler state.
- Backbone và encoder weights.
- Image size.
- Best validation mIoU.
- Tên loss.
- Số lớp và ignore index.
- Toàn bộ CLI training arguments.
- Git commit SHA khi có thể đọc được.

Nhờ đó thí nghiệm có thể tiếp tục và được truy vết chính xác hơn. Chỉ tải checkpoint từ nguồn tin cậy.

## Metric đánh giá

`metrics.py` xây confusion matrix trên toàn bộ epoch và bỏ qua target `255`. Các metric gồm:

- **Per-class IoU**: IoU riêng của từng lớp.
- **Mean IoU**: trung bình IoU trên các lớp có mặt.
- **Mean IoU without background**: mIoU từ class 1 đến 20.
- **Per-class Dice/F1** và **Mean Dice**.
- **Pixel Accuracy**: tỷ lệ tổng pixel dự đoán đúng.
- **Mean Class Accuracy**: trung bình accuracy theo lớp.
- **Confusion Matrix**: ma trận target × prediction.

`best_metrics.json` dùng `null` cho lớp không xuất hiện thay vì ghi giá trị không hợp lệ.

## Trực quan hóa

Checkpoint đã huấn luyện là bắt buộc. Encoder pretrained ImageNet không đồng nghĩa decoder segmentation đã được huấn luyện.

```bash
python visualize_predictions.py \
  --data-root data/VOC2012_train_val/VOC2012_train_val \
  --checkpoint outputs/deeplabv3plus_voc_best.pth \
  --split val \
  --indices 0 1 2 3 4 \
  --out-dir outputs/viz
```

Mỗi ảnh kết quả có bốn cột:

1. Ảnh gốc.
2. Ground-truth mask.
3. Predicted mask.
4. Overlay dự đoán trên ảnh gốc.

Vẽ learning curves sau khi training:

```bash
python plot_training_curves.py \
  --log-path outputs/train_log.csv \
  --output-path outputs/training_curves.png
```

Thêm `--show` nếu muốn mở cửa sổ Matplotlib trên máy local.

## Demo Streamlit

Khởi động ứng dụng:

```bash
python -m streamlit run streamlit_segmentation_ui.py
```

Sau đó mở địa chỉ Streamlit hiển thị trong terminal, thường là `http://localhost:8501`.

### Tab Tải ảnh thực tế

Tab này chỉ yêu cầu checkpoint, không yêu cầu tải Pascal VOC. Quy trình:

1. Nhập đường dẫn checkpoint ở sidebar.
2. Tải ảnh JPG hoặc PNG.
3. Chọn alpha overlay và ngưỡng diện tích lớp tối thiểu.
4. Bấm **Phân đoạn ảnh đã tải**.

Ứng dụng hiển thị ảnh gốc, mask màu, overlay và bảng lớp được phát hiện. Ngưỡng diện tích giúp loại lớp chỉ xuất hiện ở vài pixel nhiễu.

### Tab đánh giá VOC

Tab này cần cả checkpoint và dataset. Dataset chỉ được kiểm tra khi người dùng bấm chạy, vì vậy thiếu VOC không làm hỏng tab upload.

### Tab đồ thị huấn luyện

Đọc `outputs/train_log.csv` và hiển thị loss/mIoU của train và validation.

## Kiểm thử và CI

Chạy kiểm tra local:

```bash
python -m compileall -q .
pytest -q
```

Các unit test hiện kiểm tra:

- Metric với dự đoán hoàn hảo và void pixel.
- Confusion matrix với ví dụ có kết quả biết trước.
- Resize/pad giữ tỷ lệ và dùng `255` cho vùng padding mask.

Workflow `.github/workflows/ci.yml` chạy syntax check và pytest cho mỗi push/pull request trên Python 3.11.

## Cấu trúc repository

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── tests/
│   ├── test_dataset.py
│   └── test_metrics.py
├── config.py
├── dataset_voc.py
├── inference.py
├── metrics.py
├── plot_training_curves.py
├── streamlit_segmentation_ui.py
├── train_deeplabv3plus.py
├── visualize_predictions.py
├── voc_meta.py
├── requirements.txt
├── LICENSE
└── README.md
```

| File | Trách nhiệm |
|---|---|
| `config.py` | Hằng số số lớp, ignore index và data root |
| `dataset_voc.py` | Dataset và joint transforms ảnh–mask |
| `inference.py` | Load checkpoint, preprocess giữ tỷ lệ và dự đoán kích thước gốc |
| `metrics.py` | Confusion matrix và các metric segmentation |
| `train_deeplabv3plus.py` | Training, validation, AMP, resume và checkpoint |
| `visualize_predictions.py` | Xuất ảnh so sánh định tính |
| `plot_training_curves.py` | Vẽ loss và mIoU theo epoch |
| `streamlit_segmentation_ui.py` | Demo web cho upload ảnh và VOC |
| `voc_meta.py` | Tên lớp và Pascal VOC color map |

## Xử lý lỗi thường gặp

### Không tìm thấy dataset

```text
FileNotFoundError: ... ImageSets/Segmentation/train.txt
```

Kiểm tra `--data-root` có trỏ trực tiếp tới thư mục chứa `JPEGImages`, `SegmentationClass` và `ImageSets` hay không.

### Không tìm thấy checkpoint

Train mô hình trước hoặc tải checkpoint đã phát hành, sau đó truyền đúng đường dẫn qua `--checkpoint` hoặc sidebar Streamlit.

### CUDA out of memory

- Giảm `--batch-size` từ 8 xuống 4, 2 hoặc 1.
- Giảm `--image-size`.
- Đóng chương trình khác đang dùng GPU.
- Giữ `--amp` bật.

### DataLoader lỗi trên Windows

Thử `--num-workers 0`. Cách này chậm hơn nhưng hữu ích để xác định lỗi multiprocessing.

### PowerShell không cho activate virtual environment

Có thể chạy trực tiếp:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe train_deeplabv3plus.py
```

### Kết quả chứa nhiều lớp có diện tích rất nhỏ

Tăng **Diện tích lớp tối thiểu (%)** trong sidebar Streamlit. Đây chỉ là bộ lọc hiển thị, không thay đổi predicted mask.

## Đóng gói kết quả portfolio

Trước khi đưa dự án vào CV hoặc public GitHub, nên hoàn thành checklist:

- [ ] Chạy evaluation trên validation split cố định.
- [ ] Điền bảng kết quả bằng số liệu đã xác minh.
- [ ] Thêm biểu đồ training curves vào `assets/`.
- [ ] Thêm 5–10 qualitative predictions.
- [ ] Thêm ít nhất 2 failure cases và phân tích.
- [ ] Tạo GitHub Release chứa checkpoint hoặc link Hugging Face.
- [ ] Thay badge `OWNER/REPOSITORY`.
- [ ] Thêm repository topics và homepage demo.
- [ ] Ghi phần cứng và thời gian huấn luyện.

## Hạn chế và roadmap

### Hạn chế hiện tại

- Chưa có checkpoint công khai trong repository.
- Chưa có kết quả validation được xác minh.
- Chưa có demo online.
- Chưa benchmark latency, FPS hoặc peak VRAM.
- Chưa xuất ONNX/TensorRT.
- Pipeline hiện dùng cố định 21 lớp Pascal VOC.

### Thí nghiệm đề xuất

| Nhóm | So sánh |
|---|---|
| Loss | Cross Entropy và Cross Entropy + Dice |
| Backbone | ResNet50 và MobileNet/EfficientNet |
| Resolution | 320 và 512 |
| Augmentation | Có và không augmentation |
| Baseline | DeepLabV3+, U-Net và FCN |

Một ablation nhỏ có cùng protocol và phân tích rõ ràng có giá trị hơn nhiều kiến trúc nhưng thiếu kiểm soát thí nghiệm.

### Hướng phát triển

- Tách package thành `src/`, `scripts/`, `app/` và `configs/` khi dự án mở rộng.
- Thêm script evaluation độc lập và xuất báo cáo per-class.
- Benchmark CPU/GPU và peak VRAM.
- Xuất ONNX và tối ưu inference.
- Thêm Dockerfile.
- Deploy lên Streamlit Community Cloud hoặc Hugging Face Spaces.

## Tài liệu tham khảo

1. Liang-Chieh Chen et al., *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*, ECCV 2018.
2. Mark Everingham et al., *The Pascal Visual Object Classes Challenge: A Retrospective*, IJCV 2015.
3. Pavel Yakubovskiy, `segmentation-models-pytorch`.

Nếu sử dụng dự án cho báo cáo học thuật, hãy trích dẫn paper DeepLabV3+, Pascal VOC và các thư viện chính được sử dụng.

## License

Dự án được phát hành theo giấy phép MIT. Xem [LICENSE](LICENSE) để biết chi tiết.
