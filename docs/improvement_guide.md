# Hướng Dẫn Nâng Cấp Dự Án Theo 4 Tầng Phát Triển Phần Mềm AI

Tài liệu định hướng kiến trúc phát triển toàn diện cho nền tảng Semantic Segmentation theo 4 tầng chuẩn mực: **Problem Layer**, **AI/ML Correctness Layer**, **Software Engineering Layer**, và **Production Layer**.

---

## Tầng 1: Định Nghĩa Bài Toán & Phạm Vi (Problem Layer)

1. **Phân biệt Semantic Segmentation với Instance Segmentation & Detection**:
   - Semantic Segmentation gán nhãn ngữ nghĩa ở cấp độ pixel.
   - Không phân biệt từng thực thể riêng biệt (không vẽ bounding box, không đếm cá thể).
   - Tỷ lệ diện tích hiển thị trên UI là phần trăm pixel chiếm dụng trong ảnh, không phải là "độ tự tin" của mô hình.
2. **Quy chuẩn Input / Output**:
   - Input: Ảnh RGB tùy ý kích thước.
   - Output: Class ID mask, Mask màu Pascal VOC, Ảnh overlay blend, Bảng tỷ lệ diện tích (%) và Bản đồ bất định (Uncertainty Map).

---

## Tầng 2: Thử Nghiệm AI/ML & Đánh Giá Đa Chiều (AI/ML Correctness Layer)

### 2.1 Quy chuẩn Phân chia Dữ liệu 3 Tầng (Split Protocol)
- **Development Train (~70%)**: Huấn luyện trọng số mạng với Joint Augmentation.
- **Validation (~15%)**: Đánh giá đa chiều, tuning siêu tham số, dừng sớm (Early Stopping) và bình chọn Champion.
- **Locked Custom Holdout (~15%)**: Tập dữ liệu niêm phong, chỉ đánh giá duy nhất một lần sau khi toàn bộ cấu hình đã đóng băng.

### 2.2 Đánh giá Đa Kiến trúc (Fair Benchmarking)
- Chuẩn hóa: Cùng ResNet50 encoder, cùng kích thước ảnh, cùng hàm mất mát (CE + 0.5 Dice), cùng optimizer (AdamW) và learning rate schedule.
- So sánh 3 ứng viên độc lập:
  - Candidate A: U-Net
  - Candidate B: FPN
  - Candidate C: DeepLabV3+

### 2.3 Hệ Thống Chỉ Số Đo Lường Đa Tầng
- **Headline Metrics**: mIoU (All), mIoU (No Background), Per-Class IoU.
- **Structural Metrics**: Boundary F1 (BF-score) và Boundary IoU (đo lường độ khớp đường bao quanh contour).
- **Error Slices**: Region-size mIoU cho các vùng Small ($< 32^2$), Medium ($32^2 - 96^2$), Large ($\ge 96^2$).
- **Confusion Matrix Analysis**: Top 5 best/worst classes và các cặp nhầm lẫn thường gặp nhất (ví dụ: chair $\leftrightarrow$ background, dog $\leftrightarrow$ cat).

---

## Tầng 3: Kỹ Thuật Phần Mềm & Độ Bền Vững (Software Engineering Layer)

- **Kiểm thử tự động (Test Suite)**: Bao phủ tính toàn vẹn của dataset, chống rò rỉ (leakage), hàm metrics, logic suy luận gốc và lưu/đọc checkpoint metadata.
- **Tự động hóa CI**: Workflow GitHub Actions thực thi kiểm tra style code (Ruff), biên dịch cú pháp (`compileall`) và chạy toàn bộ unit tests (`pytest`).
- **Reproducibility Contract**: Checkpoint lưu trữ đầy đủ Git commit SHA, random seed, tham số huấn luyện và phiên bản thư viện môi trường.

---

## Tầng 4: Định Hướng Triển Khai & Vận Hành (Production Layer)

### Kiến trúc Khuyến nghị khi Mở rộng (Microservices Roadmap)
```text
Browser / Mobile Client
        ↓ (HTTP REST / WebSocket)
FastAPI Gateway Server (Input Validation, Rate Limiting, JWT)
        ↓ (Asynchronous Job Queue / Redis Stream)
Inference Workers (PyTorch / ONNX Runtime / TensorRT)
        ↓
GPU / CUDA Accelerator Cluster
```
- **Quản lý Tài nguyên**: Giới hạn 10 MB / 20 Megapixels mỗi yêu cầu để tránh tràn bộ nhớ GPU (OOM).
- **Chính sách Quyền riêng tư (Privacy Notice)**: Không lưu trữ hình ảnh người dùng vĩnh viễn trên máy chủ phục vụ suy luận.
