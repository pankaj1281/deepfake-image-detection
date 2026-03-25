"""
Streamlit web application for Fake Image Detection.

Launch with:
    streamlit run app/streamlit_app.py

Features:
  • Upload an image (JPG / PNG / BMP / WEBP)
  • Spatial + FFT prediction via the Hybrid model
  • Grad-CAM heatmap overlay
  • EXIF metadata analysis
  • Clean, responsive UI
"""

import io
import os
import sys

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow imports from the project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tensorflow as tf
from utils.data_loader import load_single_image
from utils.grad_cam import GradCAM
from utils.metadata import MetadataAnalyzer


# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load Keras model from disk (cached across reruns)."""
    return tf.keras.models.load_model(model_path)


# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────

def pil_to_numpy(pil_image: Image.Image, size: tuple) -> np.ndarray:
    """Resize and normalise a PIL image to a float32 numpy array."""
    pil_image = pil_image.convert("RGB").resize((size[1], size[0]))
    return np.array(pil_image, dtype=np.float32) / 255.0


def build_gradcam_figure(
    original: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
) -> plt.Figure:
    """Return a matplotlib figure with three panels: original / heatmap / overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap Overlay", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def fig_to_bytes(fig: plt.Figure) -> bytes:
    """Serialise a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar controls and return configuration dict."""
    st.sidebar.title("⚙️ Settings")

    # Default to a relative path; resolve it against the project root so it
    # works regardless of the working directory the app was launched from.
    _default_model_rel = os.path.join("models", "best_hybrid_model.keras")
    model_path_input = st.sidebar.text_input(
        "Model path",
        value=_default_model_rel,
        help=(
            "Relative (to project root) or absolute path to a trained "
            ".keras / .h5 model file."
        ),
    )
    # Resolve relative paths against the project root so users can type short
    # portable paths without knowing their system's absolute location.
    if not os.path.isabs(model_path_input):
        model_path = os.path.join(ROOT, model_path_input)
    else:
        model_path = model_path_input
    model_type = st.sidebar.selectbox(
        "Model type",
        options=["hybrid", "cnn"],
        index=0,
        help="Architecture of the loaded model.",
    )
    image_size = st.sidebar.selectbox(
        "Image size",
        options=[128, 224],
        index=0,
        help="Resolution the model was trained on.",
    )
    threshold = st.sidebar.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Probability above this value is classified as Fake.",
    )
    show_gradcam = st.sidebar.checkbox(
        "Show Grad-CAM heatmap", value=True
    )
    show_metadata = st.sidebar.checkbox(
        "Show EXIF metadata", value=True
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**ℹ️ First time?**  "
        "Train a model with `train.py` and then point the *Model path* field "
        "above to the saved `.keras` file."
    )

    return {
        "model_path": model_path,
        "model_type": model_type,
        "image_size": (image_size, image_size),
        "threshold": threshold,
        "show_gradcam": show_gradcam,
        "show_metadata": show_metadata,
    }


# ─────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────

def main() -> None:
    # ── Header ───────────────────────────────
    st.title("🔍 Deepfake Image Detector")
    st.markdown(
        "Upload an image to detect whether it is **Real** or **Fake** "
        "using a hybrid CNN + FFT model with Grad-CAM explainability."
    )

    cfg = render_sidebar()

    # ── Model loading ────────────────────────
    model = None
    if os.path.isfile(cfg["model_path"]):
        with st.spinner("Loading model …"):
            try:
                model = load_model(cfg["model_path"])
            except Exception as exc:
                st.error(f"Failed to load model: {exc}")
    else:
        st.warning(
            f"⚠️ **Model not found** at `{cfg['model_path']}`.\n\n"
            "A trained model file is required before predictions can be made. "
            "Follow the steps below to get started."
        )
        st.markdown("---")
        st.subheader("🚀 Quick Setup Guide")

        with st.expander("**Step 1 — Install dependencies** (if not done yet)", expanded=False):
            st.code(
                "# Create and activate a virtual environment (recommended)\n"
                "python -m venv .venv\n\n"
                "# Windows\n"
                ".venv\\Scripts\\activate\n\n"
                "# macOS / Linux\n"
                "source .venv/bin/activate\n\n"
                "# Install required packages\n"
                "pip install -r requirements.txt",
                language="bash",
            )

        with st.expander("**Step 2 — Prepare a dataset**", expanded=True):
            st.markdown(
                "Place your images in the following folders (relative to the project root):\n"
                "```\n"
                "dataset/\n"
                "  real/   ← authentic images (JPG / PNG / BMP / WEBP)\n"
                "  fake/   ← AI-generated or manipulated images\n"
                "```\n\n"
                "**Don't have a dataset yet?** Generate a small synthetic one for a quick test:\n"
            )
            st.code(
                "python scripts/create_sample_dataset.py",
                language="bash",
            )
            st.caption(
                "This creates 100 synthetic images per class so you can verify "
                "the full training pipeline before using real data."
            )

        with st.expander("**Step 3 — Train the model**", expanded=True):
            st.markdown("Run the training script **from the project root directory**:")
            st.code(
                "# Hybrid model (CNN + FFT) — recommended\n"
                "python train.py --dataset_dir dataset --model_type hybrid --epochs 20\n\n"
                "# CNN-only model (faster to train)\n"
                "python train.py --dataset_dir dataset --model_type cnn --epochs 20",
                language="bash",
            )
            st.markdown(
                "The best checkpoint is saved to **`models/best_hybrid_model.keras`** "
                "(same path as the default **Model path** in the sidebar)."
            )

        with st.expander("**Step 4 — Reload this page**", expanded=False):
            st.markdown(
                "Once training finishes, refresh this browser tab. "
                "The app will automatically load the model and be ready for predictions.\n\n"
                "If you saved the model to a different location, update the "
                "**Model path** field in the ⚙️ sidebar."
            )
        st.markdown("---")

    # ── Image upload ─────────────────────────
    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Supported formats: JPG, PNG, BMP, WEBP",
    )

    if uploaded is None:
        st.info("👆 Upload an image to get started.")
        return

    # ── Display original ─────────────────────
    pil_img = Image.open(uploaded).convert("RGB")
    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.subheader("Uploaded Image")
        st.image(pil_img, use_container_width=True)

    # ── Prediction ───────────────────────────
    if model is None:
        st.error("Cannot run prediction — model not loaded.")
        return

    with st.spinner("Analysing image …"):
        # Save upload to a temp file (needed for metadata analysis)
        import tempfile
        suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        # Prepare inputs
        spatial, fft = load_single_image(tmp_path, cfg["image_size"])

        if cfg["model_type"] == "hybrid":
            model_input = {"spatial_input": spatial, "fft_input": fft}
        else:
            model_input = spatial

        # Inference
        probability = float(model.predict(model_input, verbose=0).flatten()[0])
        label = "Fake" if probability >= cfg["threshold"] else "Real"
        confidence = probability if label == "Fake" else 1.0 - probability

    # ── Result display ───────────────────────
    with col_result:
        st.subheader("Prediction Result")

        if label == "Fake":
            st.error(f"🚨 **{label}** — Confidence: {confidence:.2%}")
        else:
            st.success(f"✅ **{label}** — Confidence: {confidence:.2%}")

        st.metric("Raw probability (Fake)", f"{probability:.4f}")

        # Confidence bar
        st.progress(probability, text=f"Fake probability: {probability:.2%}")

    # ── Grad-CAM ─────────────────────────────
    if cfg["show_gradcam"]:
        st.subheader("🔥 Grad-CAM Explainability")
        with st.spinner("Computing Grad-CAM …"):
            try:
                grad_cam = GradCAM(model)
                heatmap = grad_cam.compute_heatmap(model_input)
                original_np = np.array(
                    pil_img.resize(
                        (cfg["image_size"][1], cfg["image_size"][0])
                    ),
                    dtype=np.uint8,
                )
                overlay = grad_cam.overlay_heatmap(heatmap, original_np)
                fig = build_gradcam_figure(original_np, heatmap, overlay)
                st.pyplot(fig)

                # Download button
                st.download_button(
                    label="⬇️ Download heatmap",
                    data=fig_to_bytes(fig),
                    file_name="gradcam_heatmap.png",
                    mime="image/png",
                )
                plt.close(fig)
            except Exception as exc:
                st.warning(f"Grad-CAM failed: {exc}")

    # ── Metadata analysis ────────────────────
    if cfg["show_metadata"]:
        st.subheader("🧾 EXIF Metadata Analysis")
        meta = MetadataAnalyzer(tmp_path).summary()

        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.metric("Has EXIF data", "Yes" if meta["has_exif"] else "No")
            st.metric("Camera", meta["camera"] or "N/A")
            st.metric("Software", meta["software"])
        with meta_col2:
            st.metric("Editing software detected",
                      "Yes ⚠️" if meta["editing_detected"] else "No ✅")
            st.metric("Creation date", meta["creation_date"] or "N/A")
            score = meta["manipulation_score"]
            st.metric(
                "Metadata manipulation score",
                f"{score:.2f}",
                delta=None,
                help="Heuristic 0–1 score based on missing/suspicious metadata.",
            )

        if meta["editing_detected"]:
            st.warning(
                f"⚠️ Editing software detected: **{meta['software']}** — "
                "this image may have been digitally altered."
            )
        elif not meta["has_exif"]:
            st.info(
                "ℹ️ No EXIF metadata found — common for AI-generated or "
                "screenshot-derived images."
            )

    # ── Cleanup temp file ────────────────────
    try:
        os.unlink(tmp_path)
    except OSError:
        pass


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()
