# Hiệu chuẩn camera bằng phương pháp Zhang

**Tác giả:** Nguyễn Đức Nam Khánh

Ước lượng ma trận nội tại **K** và hệ số méo ống kính của camera từ ảnh chụp một bàn cờ phẳng, theo phương pháp của Z. Zhang (2000). Kết quả `results/camera_intrinsics.json` là đầu vào của dự án [BundleAdjustment-3DGS](../BundleAdjustment-3DGS).

## Chạy

```bash
pip install -r requirements.txt
python -m pytest -q          # 5 test trên dữ liệu tổng hợp
python calibrate.py          # data/checkerboard -> results/
```

| Tuỳ chọn | Mặc định | Ý nghĩa |
|---|---|---|
| `--images` | `data/checkerboard` | thư mục ảnh bàn cờ (đọc được `.HEIC` của iPhone, `.jpg`, `.png`) |
| `--inner-corners COLS ROWS` | `8 11` | số góc **trong** của bàn cờ (bàn 9 × 12 ô → 8 × 11 góc) |
| `--square-mm` | `17` | cạnh một ô (mm); chỉ ảnh hưởng đến đơn vị của tư thế bàn cờ, không ảnh hưởng K |
| `--outlier-factor` | `3.0` | loại ảnh có RMS lớn hơn 3 × trung vị RMS của các ảnh |
| `--fit-k3` | tắt | ước lượng cả k3 (chỉ nên bật khi bàn cờ phủ tới các góc ảnh) |
| `--output` | `results` | thư mục kết quả |

## Cấu trúc

```text
zhang_calibration/
├── calibrate.py            # điểm chạy: phát hiện góc → hiệu chuẩn → ghi JSON + hình
├── zhang/
│   ├── images.py           # đọc ảnh (HEIC/JPG/PNG), xoay theo EXIF
│   ├── corners.py          # mô tả bàn cờ, phát hiện góc dưới-pixel
│   ├── closed_form.py      # nghiệm đóng của Zhang: homography → K → tư thế
│   ├── calibration.py      # pipeline: nghiệm đóng → tinh chỉnh LM → loại ảnh ngoại lai
│   └── plots.py            # hình kết quả
├── tests/test_zhang.py     # kiểm chứng trên camera tổng hợp biết trước đáp án
├── data/checkerboard/      # 35 ảnh HEIC iPhone 15 Pro (4284 × 5712)
└── results/                # camera_intrinsics.json + figures/
```

## Phương pháp

**1. Phát hiện góc** (`corners.py`). `cv2.findChessboardCornersSB` tìm lưới trên ảnh thu nhỏ (cạnh dài 1600 px), sau đó `cv2.cornerSubPix` tinh chỉnh từng góc trên ảnh gốc 24 MP với cửa sổ bằng ¼ cạnh ô.

**2. Nghiệm đóng của Zhang** (`closed_form.py`). Với bàn cờ phẳng (Z = 0), mỗi ảnh cho một homography

$$s\,[u\;v\;1]^T = H\,[X\;Y\;1]^T,\qquad H = [h_1\;h_2\;h_3] = \lambda K\,[r_1\;r_2\;t].$$

$H$ được ước lượng bằng DLT có chuẩn hoá (Hartley). Vì $r_1 \perp r_2$ và $\|r_1\| = \|r_2\|$, với $B = K^{-T}K^{-1}$:

$$h_1^T B\,h_2 = 0,\qquad h_1^T B\,h_1 = h_2^T B\,h_2 .$$

Mỗi ảnh cho 2 phương trình tuyến tính theo 6 ẩn của $B$; xếp chồng thành $Vb = 0$ và giải bằng SVD. $K$ suy ra từ $B$ theo công thức đóng (Zhang, phụ lục B), tư thế từ $K^{-1}H$ rồi chiếu $R$ về SO(3) bằng SVD.

**3. Tinh chỉnh phi tuyến** (`calibration.py`). Từ $K$ của nghiệm đóng, Levenberg–Marquardt (`cv2.calibrateCameraExtended`) tối ưu đồng thời $K$, hệ số méo và tư thế của mọi ảnh, cực tiểu tổng bình phương sai số chiếu lại. Mô hình méo:

$$x_d = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2p_1xy + p_2(r^2 + 2x^2)$$
$$y_d = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1(r^2 + 2y^2) + 2p_2xy$$

**4. Loại ảnh ngoại lai.** Ảnh có RMS > 3 × trung vị (mờ, phát hiện sai) bị loại và bước 3 chạy lại.

### Vì sao cố định k3 = 0

Hình `coverage_distortion.png` cho thấy các góc bàn cờ chỉ phủ vùng giữa ảnh (u ≈ 600–3700, v ≈ 700–5000 px); bốn góc ảnh không có dữ liệu. Số hạng $k_3 r^6$ chỉ có tác dụng ở rìa ảnh nên dữ liệu không ràng buộc được nó. So sánh bằng kiểm định chéo 5-fold (sai số trên ảnh không dùng để hiệu chuẩn):

| Mô hình méo | RMS | RMS kiểm định chéo | Độ dịch do méo tại góc ảnh |
|---|---|---|---|
| không méo | 4.19 px | 4.22 px | 0 px |
| k1 | 1.61 px | 1.65 px | 140 px |
| **k1, k2, p1, p2 (chọn)** | **1.14 px** | **1.18 px** | **22 px** |
| k1, k2, p1, p2, k3 | 1.11 px | 1.14 px | 237 px (k3 = −2.4) |

Thêm k3 chỉ giảm 0.04 px nhưng ngoại suy ra độ méo 237 px ở góc ảnh — vùng không có điểm đo nào. Mô hình 4 hệ số là lựa chọn an toàn; Bundle Adjustment dùng cả các đặc trưng gần góc ảnh.

## Kết quả (35 ảnh iPhone 15 Pro, chế độ 35 mm, 4284 × 5712)

| Tham số | Giá trị | Độ lệch chuẩn |
|---|---|---|
| fx | 5799.86 px | ± 2.25 |
| fy | 5763.51 px | ± 2.26 |
| cx | 2147.38 px | ± 1.35 |
| cy | 2884.36 px | ± 1.71 |
| k1, k2 | 0.1915, −0.5405 | ± 0.0012, ± 0.0071 |
| p1, p2 | 0.00005, −0.00076 | |
| **RMS sai số chiếu lại** | **1.14 px** (3080 góc, 35/35 ảnh) | |

- Nghiệm đóng (không méo) cho fx = 6007, fy = 5964: lệch ~3.5% so với sau tinh chỉnh vì chưa tính méo ống kính, nhưng đủ gần để LM hội tụ.
- fx/fy = 1.0063. Ép fx = fy làm RMS tăng lên 2.6 px, nên chênh lệch này có thật trong dữ liệu. Nguyên nhân nhiều khả năng là bản in bàn cờ bị co giãn không đều theo hai chiều (máy in), hoặc bước xử lý ảnh của iPhone; với mục đích Bundle Adjustment, giữ fx, fy riêng là mô tả đúng nhất ảnh đã chụp.
- RMS 1.14 px ở 24 MP (≈ 0.8 px nếu quy về 12 MP) phân bố đều theo mọi hướng (hình `reprojection_error.png`) và không có ảnh ngoại lai — giới hạn đến từ độ phẳng của tờ giấy in và độ nét ở các hàng xa, không phải từ một ảnh hỏng.

## Hình kết quả (`results/figures/`)

| File | Nội dung |
|---|---|
| `reprojection_error.png` | RMS từng ảnh và đám sai số (dx, dy) của mọi góc |
| `coverage_distortion.png` | vị trí các góc bàn cờ trong khung hình, và độ lớn méo ống kính trên toàn ảnh |
| `detection_example.png` | ảnh có RMS trung vị: các góc phát hiện được và trục toạ độ bàn cờ |

## Đầu ra `camera_intrinsics.json`

`image_width`, `image_height`, `camera_matrix_K` (quy ước pixel OpenCV: tâm pixel ở toạ độ nguyên), `distortion_coefficients` `[k1, k2, p1, p2, k3]`, độ lệch chuẩn của từng tham số, RMS, K của nghiệm đóng, thông số bàn cờ, RMS từng ảnh và danh sách ảnh bị loại.

## Tài liệu tham khảo

1. Z. Zhang, *A flexible new technique for camera calibration*, IEEE TPAMI 22(11), 2000. https://doi.org/10.1109/34.888718
2. R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., 2004 — DLT có chuẩn hoá (chương 4).
3. OpenCV, *Camera calibration*: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
