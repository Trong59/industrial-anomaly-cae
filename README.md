# Industrial Anomaly Detection with Convolutional Autoencoder (CAE)

Dự án nghiên cứu và triển khai: **"Xây dựng và đánh giá mạng Convolutional Autoencoder trong phát hiện và khoanh vùng bất thường bề mặt sản phẩm công nghiệp"** trên tập dữ liệu MVTec AD (bắt đầu với danh mục `bottle`).

---

## 1. Mô hình triển khai & Phân chia môi trường

- **Máy cá nhân:** Phát triển mã nguồn, kiểm thử cục bộ (CPU/GPU nhỏ), quản lý Git, chạy inference/demo.
- **GitHub:** Đồng bộ mã nguồn giữa máy cá nhân và Colab (không lưu dataset/checkpoint).
- **Google Colab:** Huấn luyện (GPU), hiệu chỉnh ngưỡng (calibrate) và đánh giá toàn diện (evaluate).
- **Google Drive:** Lưu trữ lâu dài bộ dữ liệu gốc, checkpoint (`best_model.pth`, `last_checkpoint.pth`), ngưỡng và kết quả đánh giá.

---

## 2. Cấu trúc thư mục dự án

```text
industrial-anomaly-cae/
├── configs/
│   └── bottle.yaml                # Cấu hình baseline bottle (E1)
├── notebooks/
│   └── colab_train.ipynb          # Notebook thực thi trên Google Colab
├── src/
│   ├── datasets/
│   │   └── mvtec_dataset.py       # Dataset & DataLoader cho MVTec AD
│   ├── models/
│   │   └── cae.py                 # Kiến trúc mạng Convolutional Autoencoder
│   ├── engine/
│   │   ├── trainer.py              # Vòng lặp huấn luyện & validation
│   │   └── predictor.py            # Suy luận, tính anomaly score & anomaly map
│   ├── evaluation/
│   │   ├── threshold.py            # Hiệu chỉnh ngưỡng trên tập validation
│   │   └── metrics.py              # Image-level và pixel-level metrics
│   └── utils/
│       ├── seed.py                 # Cố định random seed
│       ├── config.py               # Quản lý cấu hình & thư mục kết quả
│       ├── logger.py               # Ghi log console & CSV
│       └── visualization.py        # Xuất biểu đồ và ảnh trực quan hóa
├── train.py                        # Script CLI huấn luyện
├── calibrate.py                    # Script CLI hiệu chỉnh ngưỡng
├── evaluate.py                     # Script CLI đánh giá trên test
├── predict.py                      # Script CLI dự đoán ảnh mới
├── requirements.txt                # Danh sách thư viện phụ thuộc
├── .gitignore                      # Loại trừ dataset, checkpoint, output
└── README.md
```

---

## 3. Cài đặt trên máy cá nhân

### Khởi tạo môi trường ảo Python
```bash
python -m venv .venv

# Trên Windows:
.venv\Scripts\activate

# Trên Linux/macOS:
source .venv/bin/activate

# Cài đặt thư viện:
pip install -r requirements.txt
```

---

## 4. Hướng dẫn chạy trên Google Colab

1. **Chuẩn bị Google Drive:**
   Tải bộ dữ liệu MVTec AD `bottle` và lưu vào:
   `MyDrive/CAE_Project/datasets/mvtec_anomaly_detection/bottle/`
2. **Mở notebook:**
   Mở `notebooks/colab_train.ipynb` trên Google Colab, kết nối GPU Runtime.
3. **Thực thi tuần tự:**
   - Mount Google Drive và kiểm tra GPU.
   - Clone / pull mã nguồn từ GitHub.
   - Copy dữ liệu từ Drive vào `/content/cae_data/mvtec_anomaly_detection/bottle`.
   - Huấn luyện:
     ```bash
     python train.py --config configs/bottle.yaml \
       --data-root /content/cae_data/mvtec_anomaly_detection \
       --output-root /content/drive/MyDrive/CAE_Project/outputs \
       --experiment-id E1_bottle_mse_seed42
     ```
   - Hiệu chỉnh ngưỡng:
     ```bash
     python calibrate.py --config configs/bottle.yaml \
       --data-root /content/cae_data/mvtec_anomaly_detection \
       --output-root /content/drive/MyDrive/CAE_Project/outputs \
       --experiment-id E1_bottle_mse_seed42
     ```
   - Đánh giá trên test:
     ```bash
     python evaluate.py --config configs/bottle.yaml \
       --data-root /content/cae_data/mvtec_anomaly_detection \
       --output-root /content/drive/MyDrive/CAE_Project/outputs \
       --experiment-id E1_bottle_mse_seed42
     ```

---

## 5. Chạy suy luận (Inference) trên máy cá nhân

Sau khi huấn luyện xong trên Colab, tải `best_model.pth` và file ngưỡng `bottle_thresholds.json` về máy:

```bash
python predict.py \
  --config configs/bottle.yaml \
  --checkpoint path/to/best_model.pth \
  --thresholds path/to/bottle_thresholds.json \
  --input path/to/sample_image.png \
  --output-dir results/
```
