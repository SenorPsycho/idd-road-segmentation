"""
RoadVision Nepal — Day 6 Showcase Demo
Upload an image or a short video clip -> live drivable-area overlay.

Run with:
    streamlit run app.py

Expects a checkpoint trained per the locked Phase 2 spec:
    smp.Unet(encoder_name="resnet50", encoder_weights=..., classes=2)
    512x512 input, ImageNet normalization, argmax over 2 channels
    (see Model_Train/model.py, Model_Train/config.yaml)
"""

import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# ----------------------------------------------------------------------
# Config / constants
# ----------------------------------------------------------------------

IMG_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
OVERLAY_COLOR = (57, 219, 57)  # BGR-friendly green, also fine as RGB

DEFAULT_CHECKPOINT = "runs/2026-08-11_1628/best.pt"

st.set_page_config(
    page_title="RoadVision Nepal — Drivable Area Segmentation",
    layout="wide",
)

# ----------------------------------------------------------------------
# Model loading (cached — only reload if checkpoint path/device changes)
# ----------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model checkpoint...")
def load_model(checkpoint_path: str, device: str):
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,  # weights come from the checkpoint, not ImageNet
        classes=2,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ----------------------------------------------------------------------
# Inference helpers
# ----------------------------------------------------------------------


def preprocess(frame_rgb: np.ndarray) -> torch.Tensor:
    """frame_rgb: HxWx3 uint8, RGB -> normalized 1x3x512x512 tensor."""
    resized = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    normed = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.float()


@torch.no_grad()
def predict_mask(model, frame_rgb: np.ndarray, device: str) -> np.ndarray:
    """Returns a binary drivable-area mask at the ORIGINAL frame resolution."""
    tensor = preprocess(frame_rgb).to(device)
    logits = model(tensor)  # (1, 2, 512, 512)
    pred = torch.argmax(logits, dim=1).squeeze(0).byte().cpu().numpy()  # (512, 512)
    h, w = frame_rgb.shape[:2]
    pred_full = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    return pred_full  # values in {0, 1}


def make_overlay(frame_rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    overlay = frame_rgb.copy()
    color_layer = np.zeros_like(frame_rgb)
    color_layer[:] = OVERLAY_COLOR
    drivable = mask.astype(bool)
    overlay[drivable] = (
        (1 - alpha) * frame_rgb[drivable] + alpha * color_layer[drivable]
    ).astype(np.uint8)
    return overlay


def drivable_share(mask: np.ndarray) -> float:
    return float(mask.mean()) * 100.0


# ----------------------------------------------------------------------
# Sidebar — model + display settings
# ----------------------------------------------------------------------

st.sidebar.title("Settings")

checkpoint_path = st.sidebar.text_input("Checkpoint path", value=DEFAULT_CHECKPOINT)

device_options = ["cuda"] if torch.cuda.is_available() else []
device_options.append("cpu")
device = st.sidebar.selectbox("Device", device_options, index=0)

alpha = st.sidebar.slider("Overlay opacity", 0.1, 0.9, 0.5, 0.05)

frame_skip = st.sidebar.slider(
    "Video: process every Nth frame",
    1,
    10,
    2,
    help="Higher = faster processing, coarser overlay updates. Held between processed frames.",
)

st.sidebar.caption(
    "Model: ResNet50-UNet, binary drivable-area segmentation, "
    "trained on IDD (Bangalore/Hyderabad), zero-shot on Kathmandu footage."
)

model_loaded = False
if Path(checkpoint_path).exists():
    try:
        model = load_model(checkpoint_path, device)
        model_loaded = True
        st.sidebar.success("Checkpoint loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to load checkpoint: {e}")
else:
    st.sidebar.warning("Checkpoint not found at that path yet.")

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

st.title("RoadVision Nepal")
st.caption("Drivable-area segmentation — trained on IDD, tested on Kathmandu street footage")

tab_image, tab_video = st.tabs(["Image", "Video clip"])

# --- Image tab ---------------------------------------------------------
with tab_image:
    img_file = st.file_uploader(
        "Upload a road frame (jpg/png)", type=["jpg", "jpeg", "png"], key="img_uploader"
    )

    if img_file is not None and model_loaded:
        file_bytes = np.frombuffer(img_file.read(), dtype=np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.time()
        mask = predict_mask(model, frame_rgb, device)
        infer_ms = (time.time() - t0) * 1000

        overlay = make_overlay(frame_rgb, mask, alpha)
        mask_vis = (mask * 255).astype(np.uint8)

        col1, col2, col3 = st.columns(3)
        col1.image(frame_rgb, caption="Input", use_container_width=True)
        col2.image(mask_vis, caption="Predicted mask", use_container_width=True)
        col3.image(overlay, caption="Overlay", use_container_width=True)

        st.metric("Drivable area", f"{drivable_share(mask):.1f}%")
        st.caption(f"Inference time: {infer_ms:.0f} ms on {device}")
    elif img_file is not None and not model_loaded:
        st.error("Load a valid checkpoint in the sidebar first.")

# --- Video tab -----------------------------------------------------------
with tab_video:
    vid_file = st.file_uploader(
        "Upload a short clip (mp4/mov, keep it under ~30s for a live demo)",
        type=["mp4", "mov", "avi"],
        key="vid_uploader",
    )

    if vid_file is not None and model_loaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid_file.name).suffix) as tmp_in:
            tmp_in.write(vid_file.read())
            in_path = tmp_in.name

        cap = cv2.VideoCapture(in_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = str(Path(tempfile.gettempdir()) / f"roadvision_overlay_{int(time.time())}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        progress = st.progress(0.0, text="Processing frames...")
        status = st.empty()

        last_mask = None
        frame_idx = 0
        start_time = time.time()

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if frame_idx % frame_skip == 0 or last_mask is None:
                last_mask = predict_mask(model, frame_rgb, device)

            overlay_rgb = make_overlay(frame_rgb, last_mask, alpha)
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            writer.write(overlay_bgr)

            frame_idx += 1
            if total_frames > 0:
                progress.progress(
                    min(frame_idx / total_frames, 1.0),
                    text=f"Processing frame {frame_idx}/{total_frames}",
                )

            if frame_idx % 25 == 0:
                elapsed = time.time() - start_time
                rate = frame_idx / elapsed if elapsed > 0 else 0
                status.caption(f"{rate:.1f} frames/sec on {device}")

        cap.release()
        writer.release()
        progress.empty()

        st.success(f"Done — {frame_idx} frames processed in {time.time() - start_time:.1f}s")
        st.video(out_path)

        with open(out_path, "rb") as f:
            st.download_button(
                "Download annotated video",
                data=f.read(),
                file_name="roadvision_overlay.mp4",
                mime="video/mp4",
            )
    elif vid_file is not None and not model_loaded:
        st.error("Load a valid checkpoint in the sidebar first.")

if not model_loaded:
    st.info(
        "Set the checkpoint path in the sidebar (default points at "
        f"`{DEFAULT_CHECKPOINT}`) to enable inference."
    )
