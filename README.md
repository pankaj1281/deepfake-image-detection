# Deepfake Image Detection

A **production-quality** machine learning system that detects whether an image is **Real** or **Fake** using a hybrid spatial + frequency-domain approach with an optional **pretrained EfficientNetB0 backbone**, explainable AI (Grad-CAM), and EXIF metadata analysis — served through a clean Streamlit web interface.

---

## ✨ Features

| Feature | Details |
|---|---|
| **EfficientNet Hybrid Model** | Pretrained EfficientNetB0 (ImageNet) + FFT branch + SE-attention fusion — **highest accuracy** |
| **Standard Hybrid Model** | Custom CNN (spatial) + FFT (frequency domain) feature fusion |
| **CNN Backbone** | 4-block Conv2D network with BatchNorm, Dropout, GlobalAveragePooling |
| **FFT Branch** | Log-scaled magnitude spectrum as an independent CNN branch |
| **Two-Phase Training** | Freeze backbone → train head → unfreeze → fine-tune at low LR |
| **Label Smoothing** | Reduces overconfidence and improves generalisation |
| **SE Attention** | Squeeze-and-Excitation channel attention in the fusion head |
| **Grad-CAM** | Visual explanation of _why_ the model predicted Real or Fake |
| **EXIF Analysis** | Detects missing timestamps, editing-software traces, camera info |
| **Data Augmentation** | Random flips, rotation ±20°, brightness/contrast/saturation/hue jitter, random zoom |
| **Streamlit UI** | Upload → predict → heatmap → metadata, all in the browser |
| **Batch Prediction** | Run inference on an entire directory of images |
| **Modular Code** | Clean separation into `utils/`, `models/`, `app/` packages |

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

Using a virtual environment keeps the project's dependencies isolated.

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

> After activation, your prompt will be prefixed with `(.venv)`.  
> To deactivate later, run `deactivate`.

### Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

<details>
<summary>GPU acceleration (optional, NVIDIA only)</summary>

To use an NVIDIA GPU on Linux or Windows, replace the CPU-only TensorFlow wheel with the GPU build:

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

## 🚀 Quick Start (end-to-end in 4 steps)

### 1. Prepare a dataset

#### Option A — Use your own images

Place images in the following structure (relative to the project root):

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

#### Option B — Generate synthetic test images (no download needed)

Run the helper script from the **project root**:

```bash
python scripts/create_sample_dataset.py
```

This creates 100 synthetic images per class in `dataset/real/` and `dataset/fake/`.
The images are purely synthetic — useful only to verify that the full pipeline
(training → model saving → prediction) works correctly before committing to a real dataset.

> **Note:** A model trained on synthetic data will not detect real deepfakes.
> Use a real dataset for a production-quality classifier.

### 2. Train the model

Run `train.py` from the **project root directory**:

```bash
# EfficientNet hybrid (recommended — highest accuracy)
python train.py --dataset_dir dataset --model_type efficientnet --image_size 224 --epochs 20 --fine_tune

# Hybrid model (CNN + FFT) — good accuracy, trains faster
python train.py --dataset_dir dataset --model_type hybrid --epochs 20

# CNN-only model (fastest, lower accuracy)
python train.py --dataset_dir dataset --model_type cnn --epochs 20
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

</details>

After training, the following artefacts appear in `models/`:

| File | Description |
|---|---|
| `best_efficientnet_model.keras` | Best EfficientNet checkpoint (highest val AUC) |
| `best_hybrid_model.keras` | Best hybrid checkpoint |
| `efficientnet_model_final.keras` | EfficientNet state after the last epoch |
| `training_history.png` | Accuracy & loss curves |
| `confusion_matrix.png` | Confusion matrix on the validation set |

### 3. Launch the web app

```bash
streamlit run app/streamlit_app.py
```

Navigate to `http://localhost:8501` in your browser.

> **Important:** Run this command from the **project root directory** so that the default
> model path (`models/best_hybrid_model.keras`) resolves correctly.
>
> For the EfficientNet model, set the **Model path** sidebar field to
> `models/best_efficientnet_model.keras` and **Model type** to `efficientnet`.

### 4. Predict from the CLI

**Single image:**
```bash
python predict.py \
  --model models/best_efficientnet_model.keras \
  --model_type efficientnet \
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
python main.py train  --dataset_dir dataset --model_type efficientnet --epochs 20 --fine_tune
python main.py predict --model models/best_efficientnet_model.keras --model_type efficientnet --image test.jpg
```

---

## 🌐 Streamlit Web Interface

| Panel | Description |
|---|---|
| **Uploaded Image** | Preview of the input image |
| **Prediction Result** | Real / Fake label with confidence score |
| **Grad-CAM** | Three-panel heatmap: original / heatmap / overlay |
| **EXIF Metadata** | Camera, software, creation date, manipulation score |

Use the sidebar to configure:
- **Model path** — relative (e.g. `models/best_efficientnet_model.keras`) or absolute path to a `.keras` / `.h5` file
- **Model type** — `efficientnet`, `hybrid`, or `cnn` (must match how the model was trained)
- **Image size** — resolution the model was trained on (128 or 224)
- **Decision threshold** — probability above which an image is labelled Fake
- **Show Grad-CAM / EXIF** toggles

---

## 🛠 Troubleshooting

### ⚠️ "Model not found" warning in the web app

This warning appears when no trained model exists at the configured path.
The app now shows a built-in step-by-step guide to fix this.

**Manual fix:**
1. Train a model first (see [Step 2](#2-train-the-model) above).
2. Launch the app from the **project root** so the relative path resolves:
   ```bash
   # run from the deepfake-image-detection/ directory
   streamlit run app/streamlit_app.py
   ```
3. If your model lives elsewhere, type the correct path (absolute paths work too)
   in the **Model path** sidebar field.

### Windows path separators

Windows users can use either forward slashes or backslashes in the **Model path** field:

```
models/best_efficientnet_model.keras    ✅  works on all platforms
models\best_efficientnet_model.keras    ✅  also works on Windows
C:\Users\you\models\my_model.keras      ✅  absolute path
```

### Training fails with "No supported images found"

Make sure both `dataset/real/` and `dataset/fake/` contain image files.  
Run the synthetic-data helper if you need test images quickly:

```bash
python scripts/create_sample_dataset.py
```

### `ModuleNotFoundError` when importing project modules

Always run scripts from the **project root**:

```bash
# ✅ correct
python train.py

# ❌ wrong — imports will fail
cd app
python streamlit_app.py
```

### Out-of-memory errors during training

Reduce the batch size or image size:
```bash
python train.py --batch_size 8 --image_size 128
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

The **EfficientNet Hybrid** model combines:
- **Transfer learning**: EfficientNetB0 pretrained on ImageNet provides rich visual features even with small datasets.
- **FFT frequency features**: Detects GAN/diffusion-model artifacts in the frequency domain that are invisible to the human eye.
- **SE Attention**: Squeeze-and-Excitation block learns which fused feature channels are most informative per sample.
- **Two-phase training**: Train only the head first (fast convergence), then fine-tune the whole network at a lower learning rate.
- **Label smoothing**: Prevents overconfident predictions and improves generalisation to unseen test images.

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

Training is monitored on **validation AUC** (area under the ROC curve) rather than just
accuracy — AUC is a more robust metric for imbalanced datasets and correlates better with
real-world detection performance.

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
