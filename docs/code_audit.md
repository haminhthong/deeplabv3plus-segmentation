# Báo Cáo Kiểm Tra Toàn Vẹn & Tối Giản Hóa Mã Nguồn (Code Audit)

Tài liệu ghi nhận kết quả kiểm toán mã nguồn dự án **DeepLabV3+ Semantic Segmentation Platform**, đảm bảo tính toàn vẹn AI/ML, phòng chống rò rỉ dữ liệu (data leakage), chuẩn hóa giao diện lập trình và độ tin cậy của hệ thống.

---

## 1. Thẩm Định Dữ Liệu & Rò Rỉ (Data Integrity & Anti-Leakage Audit)

### 1.1 Kiểm tra trùng lặp mã định danh (ID Duplication)
- **Kiểm toán nội bộ split (Intra-split)**: Đảm bảo không có ID ảnh nào bị xuất hiện 2 lần trong cùng một tập (train, val hoặc holdout test).
- **Kiểm toán liên split (Inter-split)**: Tuyệt đối không có giao thoa ID giữa các tập. Vi phạm sẽ kích hoạt ngoại lệ `ValueError`.

### 1.2 Thẩm tra toàn vẹn nội dung tệp bằng SHA-256
- Không chỉ dựa vào tên file (ID), pipeline áp dụng thuật toán băm SHA-256 trên từng file ảnh JPG và mask PNG.
- Ngăn chặn triệt để trường hợp cùng một nội dung ảnh được lưu dưới hai ID khác nhau trong dataset.

### 1.3 Thẩm tra cấu trúc dữ liệu hình học và nhãn ngữ nghĩa
- **Cặp ảnh và mặt nạ (Pairing)**: Mọi ID bắt buộc phải có đầy đủ ảnh gốc `JPEGImages/{id}.jpg` và mặt nạ `SegmentationClass/{id}.png`.
- **Đồng nhất kích thước (Dimension Consistency)**: Kích thước $(W, H)$ của ảnh và mask phải khớp tuyệt đối trước khi đưa vào pipeline xử lý.
- **Giá trị nhãn hợp lệ (Valid Class Range)**: Nhãn pixel mask chỉ được phép nằm trong tập $\{0, 1, \dots, 20\} \cup \{255\}$ (20 lớp VOC + 1 background + 255 ignore boundary).

---

## 2. Kiểm Toán Kiến Trúc Mã Nguồn (Architecture & Engineering Audit)

### 2.1 Loại bỏ nhầm lẫn kiến trúc Baseline FCN
- **Phát hiện**: Nhánh `--architecture fcn` trong phiên bản cũ thực chất khởi tạo `smp.FPN` (Feature Pyramid Network), gây sai lệch định danh kỹ thuật.
- **Khắc phục**: Loại bỏ tên gọi `fcn`, định danh rõ ràng 3 ứng viên độc lập:
  - `unet`: Mô hình phân đoạn U-Net chuẩn mực.
  - `fpn`: Mô hình Feature Pyramid Network cho multi-scale representation.
  - `deeplabv3plus`: Mô hình DeepLabV3+ tích hợp ASPP và Atrous Separable Convolution.

### 2.2 Sửa lỗi thiên lệch không gian trong Augmentation
- **Phát hiện**: Khi ảnh sau khi random scale nhỏ hơn kích thước target, mã nguồn cũ đệm cứng vào `pad_right` và `pad_bottom`, dẫn đến nội dung đối tượng luôn bị kéo dồn về góc trên-trái (top-left anchor bias).
- **Khắc phục**: Chuyển sang cơ chế đệm ngẫu nhiên không thiên lệch (`pad_left = random.randint(0, pad_total_w)`), giúp mô hình học được tính bất biến không gian của vật thể.

### 2.3 Phân định Transform Contract giữa Training và Serving
- **Training**: Joint random scale [0.75, 1.5] $\to$ unbiased padding $\to$ random crop $\to$ horizontal flip $\to$ affine $\to$ mild color jitter (chỉ trên ảnh RGB).
- **Validation / Serving**: Deterministic letterbox giữ nguyên tỷ lệ khung hình (aspect ratio) với đệm đều vào tâm, bảo toàn hình thái học thực tế của vật thể.

---

## 3. Bản Đồ Bất Định (Uncertainty & Reliability Quantification)

- Suy luận tại độ phân giải gốc thông qua nội suy Logits song tuyến trước khi áp dụng hàm Softmax.
- Xuất bản đồ Normalized Entropy:
  $$H(x) = -\frac{1}{\ln(C)} \sum_{c=0}^{C-1} P(c \mid x) \ln(P(c \mid x) + \epsilon) \in [0, 1]$$
- Trực quan hóa các vùng ranh giới phức tạp hoặc các điểm mô hình chưa chắc chắn, nâng cao tính minh bạch và an toàn khi ứng dụng thực tế.
