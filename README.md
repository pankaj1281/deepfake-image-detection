# Fake Image Detection (Deepfake Detection)

A **production-quality** machine learning system that detects whether an image is **Real** or **Fake** using a hybrid spatial + frequency-domain approach, explainable AI (Grad-CAM), and EXIF metadata analysis — served through a clean Streamlit web interface.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Hybrid Model** | CNN (spatial) + FFT (frequency domain) feature fusion |
| **CNN Backbone** | 4-block Conv2D network with BatchNorm, Dropout, GlobalAveragePooling |
| **FFT Branch** | Log-scaled magnitude spectrum as an independent CNN branch |
| **Grad-CAM** | Visual explanation of _why_ the model predicted Real or Fake |
| **EXIF Analysis** | Detects missing timestamps, editing-software traces, camera info |
| **Data Augmentation** | Random flips, rotation, brightness/contrast jitter via `tf.data` |
| **Streamlit UI** | Upload → predict → heatmap → metadata, all in the browser |
| **Batch Prediction** | Run inference on an entire directory of images |
| **Modular Code** | Clean separation into `utils/`, `models/`, `app/` packages |

---

## 🗂 Project Structure

```
deepfake-image-detection/
├── dataset/
│   ├── real/          ← real images
│   └── fake/          ← fake / AI-generated images
├── models/            ← saved model checkpoints & plots
├── utils/
│   ├── __init__.py
│   ├── data_loader.py ← dataset loading, FFT features, augmentation
│   ├── grad_cam.py    ← Grad-CAM implementation
│   └── metadata.py   ← EXIF metadata analysis
├── app/
│   └── streamlit_app.py ← Streamlit web interface
├── notebooks/         ← Jupyter notebooks (exploratory work)
├── main.py            ← Unified CLI entry point
├── train.py           ← Full training pipeline
├── predict.py         ← Inference script (single image or batch)
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

### CNN Model (spatial only)
```
Input (H×W×3)
  → Conv2D(32) → BN → ReLU → MaxPool
  → Conv2D(64) → BN → ReLU → MaxPool
  → Conv2D(128) → BN → ReLU → MaxPool
  → Conv2D(256) → BN → ReLU → MaxPool
  → GlobalAveragePooling → Dense(256) → Dropout → Dense(128) → Dropout
  → Dense(1, sigmoid)
```

### Hybrid Model (spatial + frequency)
```
Spatial branch (H×W×3)          FFT branch (H×W×1)
  4-block Conv2D + GAP    ┐         3-block Conv2D + GAP
                          └──Concatenate──┐
                                          ↓
                              Dense(512) → Dropout
                              Dense(256) → Dropout
                              Dense(1, sigmoid)
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the Dataset

Place your images in the expected structure:

```
dataset/
  real/   ← real images (JPG / PNG / BMP / WEBP)
  fake/   ← fake / AI-generated images
```

### 3. Train the Model

```bash
# Hybrid model (CNN + FFT) — recommended
python train.py --dataset_dir dataset --model_type hybrid --epochs 20

# CNN-only model
python train.py --dataset_dir dataset --model_type cnn --epochs 20
```

Training options:

| Flag | Default | Description |
|---|---|---|
| `--dataset_dir` | `dataset` | Root dataset directory |
| `--model_type` | `hybrid` | `hybrid` or `cnn` |
| `--image_size` | `128` | Resize to N×N pixels |
| `--epochs` | `20` | Max training epochs |
| `--batch_size` | `32` | Batch size |
| `--output_dir` | `models` | Where to save model & plots |
| `--no_augment` | — | Disable data augmentation |

After training, saved artefacts appear in `models/`:
- `best_hybrid_model.keras` — best checkpoint (lowest val loss)
- `hybrid_model_final.keras` — model after last epoch
- `training_history.png` — accuracy & loss curves
- `confusion_matrix.png` — confusion matrix

### 4. Run Predictions

**Single image:**
```bash
python predict.py \
  --model models/best_hybrid_model.keras \
  --model_type hybrid \
  --image path/to/image.jpg
```

**Batch (entire directory):**
```bash
python predict.py \
  --model models/best_hybrid_model.keras \
  --input_dir path/to/images/ \
  --output_dir predictions/
```

**Using the unified CLI:**
```bash
python main.py predict --model models/best_hybrid_model.keras --image test.jpg
python main.py train --dataset_dir dataset --epochs 20
```

### 5. Launch the Web App

```bash
streamlit run app/streamlit_app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 🌐 Streamlit Web Interface

| Panel | Description |
|---|---|
| **Uploaded Image** | Preview of the input image |
| **Prediction Result** | Real / Fake label with confidence score |
| **Grad-CAM** | Three-panel heatmap: original / heatmap / overlay |
| **EXIF Metadata** | Camera, software, creation date, manipulation score |

Use the sidebar to configure the model path, image size, decision threshold, and which panels to display.

---

## 🔥 Grad-CAM Explainability

Grad-CAM highlights the image regions responsible for the prediction:

- **Red / yellow** areas → high influence on the _Fake_ decision
- **Blue / purple** areas → low influence

The heatmap is computed from the last Conv2D layer of the spatial branch and overlaid on the original image with 40 % transparency.

---

## 🧾 EXIF Metadata Scoring

The `MetadataAnalyzer` class computes a heuristic **manipulation score** (0 – 1):

| Condition | Score added |
|---|---|
| No EXIF data at all | +0.30 |
| Editing software detected (Photoshop, GIMP, etc.) | +0.30 |
| No camera make / model | +0.20 |
| No creation date | +0.20 |

---

## 📊 Evaluation Metrics

After training, the script prints:
- Accuracy, Precision, Recall, F1-score (per class)
- Confusion matrix (saved as PNG)
- Training vs. validation accuracy / loss curves (saved as PNG)

---

## 📦 Dependencies

```
tensorflow>=2.10.0
opencv-python>=4.7.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
streamlit>=1.20.0
Pillow>=9.4.0
seaborn>=0.12.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🗒 Notebooks

The `notebooks/` directory is reserved for exploratory Jupyter notebooks (EDA, prototype models, visualisations).

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Add my feature"`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

This project is released under the MIT License.