# Deepfake Image Detection using Optimized CNN

A **production-quality** machine learning system that detects whether an image is **Real** or **Fake** using a hybrid spatial + frequency-domain approach with an optional **pretrained EfficientNetB0 backbone**, explainable AI (Grad-CAM), and EXIF metadata analysis — served through a clean Streamlit web interface.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Fast Training** | Lightweight EfficientNetB0 backbone with two-phase training; 128×128 images finish initial training in under 30 minutes on a modern GPU with a 10 k-image dataset |
| **Balanced Prediction** | Automatic class-weight computation eliminates single-class prediction bias |
| **Real/Fake Classification with Confidence** | Every prediction returns a label (**Real** / **Fake**) plus a confidence score (%) |
| **EfficientNet Hybrid Model** | Pretrained EfficientNetB0 (ImageNet) + FFT branch + SE-attention fusion — **highest accuracy** |
| **Standard Hybrid Model** | Custom CNN (spatial) + FFT (frequency domain) feature fusion |
| **CNN Backbone** | 4-block Conv2D network with BatchNorm, Dropout, GlobalAveragePooling |
| **FFT Branch** | Log-scaled magnitude spectrum as an independent CNN branch |
| **Two-Phase Training** | Freeze backbone → train head → unfreeze → fine-tune at low LR |
| **Label Smoothing** | Reduces overconfidence and improves generalisation |
| **SE Attention** | Squeeze-and-Excitation channel attention in the fusion head |
| **Mixed Precision** | Automatically enabled on GPU for ~2× throughput (float16 compute, float32 weights) |
| **Class Imbalance Handling** | `compute_class_weight("balanced")` passed to `model.fit()` |
| **Grad-CAM** | Visual explanation of _why_ the model predicted Real or Fake |
| **EXIF Analysis** | Detects missing timestamps, editing-software traces, camera info |
| **Data Augmentation** | Random flips, rotation ±20°, brightness/contrast/saturation/hue jitter, random zoom |
| **Streamlit UI** | Upload → predict → heatmap → metadata, all in the browser |
| **Batch Prediction** | Run inference on an entire directory of images |
| **Modular Code** | Clean separation into `utils/`, `models/`, `app/` packages |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Deep learning | TensorFlow / Keras ≥ 2.10 |
| Transfer learning | EfficientNetB0 (ImageNet pretrained) |
| Data pipeline | `tf.data` with cache & prefetch |
| Image processing | OpenCV, Pillow |
| Classical ML | scikit-learn (class weights, metrics, confusion matrix) |
| Visualisation | Matplotlib, Seaborn |
| Web app | Streamlit |
| Metadata | Pillow EXIF |

---

## 🗂 Dataset Structure

```
dataset/
  real/   ← authentic images (JPG / PNG / BMP / WEBP)
  fake/   ← AI-generated or manipulated images
```

Both sub-folders must contain at least a few images.

**Suggested public datasets:**

| Dataset | Description | Link |
|---|---|---|
| FaceForensics++ | Video-frame manipulations | [GitHub](https://github.com/ondyari/FaceForensics) |
| DFDC (Kaggle) | DeepFake Detection Challenge | [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge) |
| 140k Real vs Fake | Real Flickr photos + StyleGAN2 fakes | [Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) |

---

## 🗂 Project Structure

```
deepfake-image-detection/
├── dataset/
│   ├── real/          ← real images (JPG / PNG / BMP / WEBP)
│   └── fake/          ← fake / AI-generated images
├── models/            ← saved model checkpoints & plots (created after training)
├── scripts/
│   └── create_sample_dataset.py  ← generate synthetic images for pipeline testing
├── utils/
│   ├── __init__.py
│   ├── data_loader.py ← dataset loading, FFT features, tf.data pipeline, augmentation
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

## 🛠 Installation

### Requirements

- **Python 3.8 – 3.11** (TensorFlow 2.x does not yet support Python 3.12)
- **pip** 22 or later
- **Git**
- Optional: an NVIDIA GPU with CUDA 11.x / cuDNN 8.x for faster training

### Step 1 — Clone the repository

```bash
git clone https://github.com/pankaj1281/deepfake-image-detection.git
cd deepfake-image-detection
```

### Step 2 — Create a virtual environment (recommended)

**Windows (Command Prompt / PowerShell):**
```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

<details>
<summary>GPU acceleration (optional, NVIDIA only)</summary>

```bash
pip install tensorflow[and-cuda]   # TensorFlow ≥ 2.13 on Linux/Windows
```

Verify GPU detection:
```python
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

> **macOS (Apple Silicon):** Use `tensorflow-metal` instead:
> ```bash
> pip install tensorflow-metal
> ```

</details>

---

## 🚀 How to Train

### Prepare a dataset

Place images in `dataset/real/` and `dataset/fake/`, or generate synthetic test images:

```bash
python scripts/create_sample_dataset.py
```

### Train the model

```bash
python train.py
```

The default command above trains the hybrid CNN+FFT model on the `dataset/` folder with sensible defaults (128×128, 20 epochs, batch size 32, augmentation enabled).

#### More options

```bash
# EfficientNet hybrid (recommended — highest accuracy)
python train.py --dataset_dir dataset --model_type efficientnet --image_size 224 --epochs 20 --fine_tune

# Hybrid model (CNN + FFT) — good accuracy, trains faster
python train.py --dataset_dir dataset --model_type hybrid --epochs 20

# CNN-only model (fastest, lower accuracy)
python train.py --dataset_dir dataset --model_type cnn --epochs 20

# Adjust decision threshold at evaluation time (default 0.5)
python train.py --threshold 0.4
```

<details>
<summary>All training flags</summary>

| Flag | Default | Description |
|---|---|---|
| `--dataset_dir` | `dataset` | Root dataset directory |
| `--model_type` | `hybrid` | `efficientnet`, `hybrid`, or `cnn` |
| `--image_size` | `128` | Resize to N×N pixels (use `224` for EfficientNet) |
| `--epochs` | `20` | Max training epochs (phase 1) |
| `--batch_size` | `32` | Batch size |
| `--output_dir` | `models` | Where to save model & plots |
| `--no_augment` | — | Disable data augmentation |
| `--fine_tune` | — | Enable EfficientNet backbone fine-tuning (phase 2) |
| `--fine_tune_epochs` | `10` | Additional fine-tuning epochs |
| `--threshold` | `0.5` | Decision threshold for binary classification |

</details>

After training, the following artefacts appear in `models/`:

| File | Description |
|---|---|
| `best_efficientnet_model.keras` | Best EfficientNet checkpoint (highest val AUC) |
| `best_hybrid_model.keras` | Best hybrid checkpoint |
| `efficientnet_model_final.keras` | EfficientNet state after the last epoch |
| `training_history.png` | Accuracy & loss curves |
| `confusion_matrix.png` | Confusion matrix on the validation set |

---

## 🔍 How to Predict

```bash
python predict.py --image path/to/image.jpg --model models/best_hybrid_model.keras
```

#### Batch prediction (entire directory)

```bash
python predict.py \
  --model models/best_hybrid_model.keras \
  --input_dir path/to/images/ \
  --output_dir predictions/
```

#### Using the EfficientNet model

```bash
python predict.py \
  --model models/best_efficientnet_model.keras \
  --model_type efficientnet \
  --image path/to/image.jpg
```

#### All prediction flags

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Path to saved Keras model |
| `--model_type` | `hybrid` | `efficientnet`, `hybrid`, or `cnn` |
| `--image` | — | Single image path |
| `--input_dir` | — | Directory for batch prediction |
| `--output_dir` | `predictions` | Where to save Grad-CAM heatmaps |
| `--image_size` | `128` | Must match training size |
| `--threshold` | `0.5` | Adjustable decision threshold |
| `--no_gradcam` | — | Skip Grad-CAM heatmap generation |

---

## 📊 Sample Output

```
═══════════════════════════════════════════════════════
  Image      : test_face.jpg
  Prediction : Fake
  Confidence : 94.32%
  Raw Prob.  : 0.9432
───────────────────────────────────────────────────────
  Has EXIF   : False
  Camera     : None
  Software   : None
  Meta Score : 0.50
═══════════════════════════════════════════════════════
```

---

## 📈 Performance Metrics

After training, the script prints class distribution, predictions distribution, and a full classification report:

```
──────────────────────────────────────────────────
  Class distribution (training set):
    Real (0) :    800  (50.0%)
    Fake (1) :    800  (50.0%)
    Total    :   1600
──────────────────────────────────────────────────

  Predictions distribution (validation set):
    Predicted Real (0) :    192
    Predicted Fake (1) :    208
    Threshold used     :    0.5
──────────────────────────────────────────────────

  Validation Accuracy : 0.9250
  Weighted F1-Score   : 0.9248
──────────────────────────────────────────────────

              precision    recall  f1-score   support

        Real       0.93      0.92      0.92       200
        Fake       0.92      0.93      0.93       200

    accuracy                           0.93       400
   macro avg       0.93      0.93      0.93       400
weighted avg       0.93      0.93      0.93       400
```

Training curves and a confusion matrix PNG are automatically saved to the `models/` directory.

---

## 🌐 Streamlit Web Interface

```bash
streamlit run app/streamlit_app.py
```

Navigate to `http://localhost:8501` in your browser.

| Panel | Description |
|---|---|
| **Uploaded Image** | Preview of the input image |
| **Prediction Result** | Real / Fake label with confidence score |
| **Grad-CAM** | Three-panel heatmap: original / heatmap / overlay |
| **EXIF Metadata** | Camera, software, creation date, manipulation score |

---

## 🔥 Grad-CAM Explainability

Grad-CAM highlights the image regions responsible for the prediction:

- **Red / yellow** areas → high influence on the _Fake_ decision
- **Blue / purple** areas → low influence

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

### EfficientNet Hybrid Model ⭐ (recommended)
```
Spatial branch (H×W×3)                FFT branch (H×W×1)
  EfficientNetB0 (ImageNet weights)       4-block Conv2D + GAP
  → GlobalAveragePooling                  → Dense(256)
  → Dense(256) → Dropout           ┐
                                   └── Concatenate ──┐
                                                      ↓
                                       Dense(512) → BN → Dropout
                                       SE Attention (channel-wise)
                                       Dense(256) → Dropout
                                       Dense(1, sigmoid)
```

---

## 🚀 Future Improvements

- [ ] Add MobileNetV3 / EfficientNetV2 backbone options for even faster inference
- [ ] Video deepfake detection (frame-level + temporal fusion)
- [ ] Online hard example mining (OHEM) for better handling of difficult samples
- [ ] Quantization & TFLite export for edge/mobile deployment
- [ ] REST API endpoint (`FastAPI`) for production serving
- [ ] Ensemble of EfficientNet + ViT (Vision Transformer) predictions
- [ ] Active learning loop to continuously improve with user-labelled corrections

---

## 🛠 Troubleshooting

### ⚠️ "Model not found" warning in the web app

Train a model first, then launch Streamlit from the **project root**:
```bash
python train.py
streamlit run app/streamlit_app.py
```

### Training fails with "No supported images found"

Make sure both `dataset/real/` and `dataset/fake/` contain image files:
```bash
python scripts/create_sample_dataset.py
```

### Out-of-memory errors during training

Reduce the batch size or image size:
```bash
python train.py --batch_size 8 --image_size 128
```

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Add my feature"`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

This project is released under the MIT License.

