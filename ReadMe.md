# Hướng dẫn chạy mô hình phân đoạn ảnh DeepLabV3+

Dự án này huấn luyện mô hình `DeepLabV3+` (encoder `ResNet50`) trên bộ dữ liệu Pascal VOC 2012 chưa có sẵn trong thư mục `data/`. Thầy giải nén file data.zip và bỏ vào Soure để chạy ạ.

### 1. Cấu trúc lưu trữ trong project

```text
Soure_Nhom02_PhanDoanAnh(DeepLab)/
├─ data/
│  ├─ VOC2012_train_val/VOC2012_train_val/
│  └─ VOC2012_test/VOC2012_test/
├─ outputs/                      # sinh ra sau khi train/visualize
│  ├─ deeplabv3plus_voc_best.pth
│  ├─ train_log.csv
│  ├─ training_curves.png
│  └─ viz/
├─ config.py                     # hằng số chung: NUM_CLASSES, IGNORE_INDEX, VOC_ROOT
├─ voc_meta.py                   # tên lớp VOC + colormap VOC
├─ dataset_voc.py                # Dataset Pascal VOC và transform
├─ train_deeplabv3plus.py        # train DeepLabV3+
├─ visualize_predictions.py      # vẽ Image/GT/Prediction/Overlay
├─ plot_training_curves.py       # vẽ đồ thị loss và mIoU
├─ streamlit_segmentation_ui.py  # giao diện người dùng Streamlit
├─ requirements.txt
└─ Hướng dẫn chạy.md
```

### 2. Vai trò từng file `.py`

- `config.py`: cấu hình lõi dùng lại cho toàn bộ pipeline.
- `voc_meta.py`: định nghĩa 21 lớp Pascal VOC và bảng màu hiển thị.
- `dataset_voc.py`: đọc ảnh/mask VOC (`JPEGImages`, `SegmentationClass`) + transform train/val.
- `train_deeplabv3plus.py`: huấn luyện mô hình `segmentation_models_pytorch.DeepLabV3Plus`, lưu checkpoint tốt nhất.
- `visualize_predictions.py`: nạp checkpoint và xuất ảnh trực quan 4 cột.
- `plot_training_curves.py`: đọc `outputs/train_log.csv` và vẽ đồ thị huấn luyện.
- `streamlit_segmentation_ui.py`: UI để upload ảnh thật, dự đoán phân đoạn, liệt kê lớp đối tượng, xem đồ thị.

### 3. Cấu trúc dữ liệu VOC đang dùng

Mặc định script sử dụng thư mục:

`data/VOC2012_train_val/VOC2012_train_val`

Trong đó phải có các thư mục/file con:

- `JPEGImages/`
- `SegmentationClass/`
- `ImageSets/Segmentation/train.txt`
- `ImageSets/Segmentation/val.txt`

Nếu thầy giữ nguyên bộ dữ liệu VOC như đã giải nén thì không cần chỉnh gì thêm.

### 4. Cài thư viện (Windows)

```bash
pip install -r requirements.txt
```

### 5. Huấn luyện mô hình

```bash
python train_deeplabv3plus.py --epochs 40 --batch-size 8 --image-size 320
```

Kết quả được lưu trong thư mục `outputs/`:

- `deeplabv3plus_voc_best.pth`: checkpoint tốt nhất theo mIoU trên tập `val`
- `train_log.csv`: log loss và mIoU theo từng epoch

Có thể thay đổi số epoch hoặc batch size nếu muốn:

```bash
python train_deeplabv3plus.py --epochs 30 --batch-size 4
```

### 6. Trực quan hóa kết quả phân đoạn

Sau khi đã có checkpoint (đã train xong), chạy:

```bash
python visualize_predictions.py --indices 0 1 2 3 4 --checkpoint outputs/deeplabv3plus_voc_best.pth
```

Script sẽ:

- Đọc checkpoint trong `outputs/deeplabv3plus_voc_best.pth`
- Dự đoán trên các chỉ số mẫu của tập `val`
- Vẽ 4 cột: `Image` / `Ground truth` / `Prediction` / `Overlay`
- Lưu từng ảnh trực quan vào `outputs/viz/`

### 7. Trực quan hóa đồ thị quá trình huấn luyện

Sau khi train xong, file log nằm trong `outputs/train_log.csv`. Thầy có thể vẽ đồ thị:

```bash
python plot_training_curves.py --log-path outputs/train_log.csv --output-path outputs/training_curves.png
```

Script sẽ tạo ảnh `outputs/training_curves.png` gồm 2 biểu đồ:

- Loss theo epoch (Train/Val)
- mIoU theo epoch (Train/Val)

### 8. Giao diện người dùng (Streamlit)

Nếu thầy muốn xem dự đoán và đồ thị ngay trong một giao diện web:

```bash
python -m streamlit run streamlit_segmentation_ui.py
```

Sau đó mở đường link mà Streamlit in ra (thường là `http://localhost:8501`).

Các chức năng chính trong UI:

- Upload ảnh thực tế (`jpg/png`) để phân đoạn.
- Hiển thị `Input` / `Prediction (Mask)` / `Overlay`.
- Tự liệt kê các lớp đối tượng xuất hiện (20 lớp đối tượng VOC, không tính background).
- Tab xem đồ thị loss và mIoU theo epoch.

### 9. Tùy chỉnh nhanh

- **Đổi backbone**: thêm tham số, ví dụ `--encoder resnet101`
- **Đổi kích thước ảnh**: `--image-size 512`
- **Đổi đường dẫn data** (nếu không để đúng như mặc định): `--data-root <duong_dan_toi_VOC>`

### 10. Ghi chú

- Dataset VOC dùng nhãn `255` làm ignore index, script đã xử lý sẵn.
- Nếu máy có GPU CUDA, PyTorch sẽ tự động dùng GPU, nếu không sẽ chạy trên CPU (sẽ chậm hơn).
