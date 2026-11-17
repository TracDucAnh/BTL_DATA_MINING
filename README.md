# Đề tài 07: Chẩn đoán Sớm Bệnh lý Mãn tính Đái tháo đường & Tim mạch

> Healthcare Predictive Analytics — Bài tập lớn môn Khai phá dữ liệu (Data Mining)

| Mục | Nội dung |
|---|---|
| Tác vụ khai phá chính | Phân loại y tế (Medical Classification & Risk Scoring) |
| Chuẩn đầu ra (CLO) | L.O.3.2, L.O.4.1, L.O.4.2, L.O.4.3 |
| Tập dữ liệu | Pima Indians Diabetes Database / Heart Disease UCI Dataset |

## Mục tiêu bài toán

Sàng lọc nguy cơ mắc bệnh tiểu đường hoặc bệnh mạch vành dựa trên các chỉ số sinh hóa và thói quen sinh hoạt cơ bản.

## Các bước thực hiện

1. **Tiền xử lý** — Phát hiện các giá trị sinh học phi thực tế (huyết áp, glucose, insulin, BMI bằng 0), thay thế bằng ước lượng imputation (KNNImputer / MICE); chuẩn hóa dữ liệu bằng StandardScaler.
2. **Mô hình hóa** — Xây dựng các mô hình Naive Bayes, Decision Tree (ID3/CART), SVM (kernel RBF) và Multi-layer Perceptron.
3. **Đánh giá** — Phân tích ROC-AUC, Specificity và Sensitivity (ưu tiên Sensitivity để không bỏ sót người mắc bệnh).
4. **Ứng dụng** — Giao diện hỗ trợ nhân viên y tế nhập chỉ số xét nghiệm, xuất báo cáo nguy cơ kèm biểu đồ radar thể hiện các chỉ số vượt ngưỡng cảnh báo.

## Cấu trúc thư mục

```
.
├── data/                    # Dữ liệu
│   ├── raw/                 #   Dữ liệu gốc tải về (không chỉnh sửa)
│   └── processed/           #   Dữ liệu đã làm sạch, imputation, chuẩn hóa
│
├── eda_preprocessing/       # Bước 1: Khám phá & tiền xử lý dữ liệu
│   ├── eda.py               #   Script EDA
│   ├── preprocessing.py     #   Script tiền xử lý
│   └── figures/             #   Biểu đồ sinh ra trong quá trình EDA
│
├── model/                   # Bước 2 + 3: Huấn luyện & đánh giá mô hình
│   ├── train.py             #   Script huấn luyện
│   ├── evaluate.py          #   Script đánh giá
│   ├── saved_models/        #   Mô hình + scaler/imputer đã huấn luyện
│   └── reports/             #   Bảng metric, ROC curve, confusion matrix
│
├── app/                     # Bước 4: Ứng dụng web
│   ├── backend/             #   API nạp mô hình và trả kết quả dự đoán
│   └── frontend/            #   Giao diện HTML / CSS / JavaScript
│       ├── css/
│       ├── js/
│       └── assets/
│
└── README.md
```

Mỗi thư mục chính có `README.md` riêng mô tả chi tiết nội dung và quy ước. Các file `.gitkeep` chỉ để Git giữ lại thư mục rỗng, có thể xóa khi thư mục đã có file.

## Luồng làm việc

```
data/raw ──► eda_preprocessing ──► data/processed ──► model ──► model/saved_models ──► app
```

## Thành viên nhóm

| Họ tên | MSSV | Phụ trách |
|---|---|---|
| | | |
