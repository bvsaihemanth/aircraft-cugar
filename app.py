import time
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = r"C:\Users\bomma\Desktop\pp1\cugar_jets_best.pth"

IMAGE_SIZE = 224
NUM_TOP_PREDICTIONS = 5

IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32
).view(3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32
).view(3, 1, 1)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CUGAR Aircraft Recognition",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        margin-bottom: 20px;
    }

    .prediction-name {
        font-size: 30px;
        font-weight: 700;
    }

    .confidence {
        font-size: 20px;
        margin-top: 8px;
    }

    .metric-card {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #ddd;
        text-align: center;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 700;
    }

    .metric-label {
        font-size: 14px;
        color: #777;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CUGAR ViT BACKBONE
# Exact architecture from evaluation script
# ============================================================

class ViTB16Backbone(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        net = models.vit_b_16(weights=None)

        self.conv_proj = net.conv_proj
        self.class_token = net.class_token
        self.encoder = net.encoder

        self.dropout = nn.Dropout(p=0.1)

        self.fc = nn.Linear(
            768,
            num_classes
        )

    def _encode(self, x):

        B = x.shape[0]

        x = self.conv_proj(x)

        x = x.flatten(2).transpose(1, 2)

        cls = self.class_token.expand(B, -1, -1)

        x = torch.cat(
            [cls, x],
            dim=1
        )

        return self.encoder(x)

    def forward(self, x):

        tokens = self._encode(x)

        return self.fc(
            self.dropout(tokens[:, 0])
        )

    def forward_with_feat(self, x):

        tokens = self._encode(x)

        cls = tokens[:, 0]

        patches = tokens[:, 1:]

        B = x.shape[0]

        feat = patches.reshape(
            B,
            14,
            14,
            768
        ).permute(
            0,
            3,
            1,
            2
        ).contiguous()

        logits = self.fc(
            self.dropout(cls)
        )

        return feat, cls, logits


# ============================================================
# TAMRZ
# ============================================================

class TAMRZModule(nn.Module):

    def __init__(
        self,
        num_classes,
        zoom_ratios=(0.60, 0.35)
    ):
        super().__init__()

        self.zoom_ratios = zoom_ratios

        self.attn_conv = nn.Conv2d(
            768,
            1,
            kernel_size=1
        )

        self.attn_head = nn.Linear(
            768,
            num_classes
        )

        self.zoom_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        self.level_weights = nn.Parameter(
            torch.tensor([0.50, 0.30, 0.20])
        )

    def get_attn(self, feat):

        a = F.relu(
            self.attn_conv(feat)
        ).squeeze(1)

        B, H, W = a.shape

        flat = a.view(B, -1)

        return (
            flat /
            (flat.max(1, keepdim=True)[0] + 1e-8)
        ).view(B, H, W)

    def _zoom_crops(
        self,
        images,
        attn
    ):

        B, C, H, W = images.shape

        attn_up = F.interpolate(
            attn.unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        crops_list = []

        for ratio in self.zoom_ratios:

            ch = int(H * ratio)
            cw = int(W * ratio)

            crops = []

            for b in range(B):

                fi = attn_up[b].argmax()

                cy = int(fi // W)
                cx = int(fi % W)

                y1 = min(
                    max(0, cy - ch // 2),
                    H - ch
                )

                x1 = min(
                    max(0, cx - cw // 2),
                    W - cw
                )

                crop = images[
                    b:b + 1,
                    :,
                    y1:y1 + ch,
                    x1:x1 + cw
                ]

                crop = F.interpolate(
                    crop,
                    (224, 224),
                    mode="bilinear",
                    align_corners=False
                )

                crops.append(crop)

            crops_list.append(
                torch.cat(crops, dim=0)
            )

        return crops_list

    def forward(
        self,
        backbone,
        feat,
        pooled,
        logits_g,
        images
    ):

        attn = self.get_attn(feat)

        zoom_crops_list = self._zoom_crops(
            images,
            attn
        )

        zoom_logits = [
            self.zoom_head(
                backbone._encode(zc)[:, 0]
            )
            for zc in zoom_crops_list
        ]

        w = F.softmax(
            self.level_weights,
            dim=0
        )

        fused = (
            w[0] * logits_g
            +
            sum(
                w[i + 1] * zl
                for i, zl in enumerate(zoom_logits)
            )
        )

        return fused


# ============================================================
# SPATIAL UNCERTAINTY HEAD
# ============================================================

class SpatialUncertaintyHead(nn.Module):

    def __init__(
        self,
        in_channels=768,
        num_classes=88,
        dropout_p=0.15
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=1
        )

    def forward(self, feat):

        return self.conv(feat)


# ============================================================
# INTERVENTION GENERATOR
# ============================================================

class InterventionGenerator(nn.Module):

    def __init__(
        self,
        use_inpainting=False
    ):
        super().__init__()

        n_types = (
            3
            if use_inpainting
            else 2
        )

        self.w = nn.Parameter(
            torch.ones(n_types) / n_types
        )


# ============================================================
# COMPLETE CUGAR ViT
# ============================================================

class CUGARViT(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.backbone = ViTB16Backbone(
            num_classes
        )

        self.spatial = SpatialUncertaintyHead(
            768,
            num_classes,
            0.15
        )

        self.tamrz = TAMRZModule(
            num_classes
        )

        self.intervener = InterventionGenerator(
            use_inpainting=False
        )

        self.register_buffer(
            "class_weights",
            torch.ones(num_classes)
        )

        self.register_buffer(
            "logit_adjustment",
            torch.zeros(num_classes)
        )

    def forward(self, x):

        feat, pooled, logits_g = (
            self.backbone.forward_with_feat(x)
        )

        return self.tamrz(
            self.backbone,
            feat,
            pooled,
            logits_g,
            x
        )


# ============================================================
# DEVICE
# ============================================================

@st.cache_resource
def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


DEVICE = get_device()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    checkpoint_path = Path(CHECKPOINT_PATH)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    classes = checkpoint["classes"]

    model = CUGARViT(
        num_classes=len(classes)
    )

    try:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    except RuntimeError:

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=False
        )

    model = model.to(DEVICE)

    model.eval()

    return model, classes, checkpoint


# ============================================================
# IMAGE PREPROCESSING
# Matches evaluation code
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    # Same first resize as evaluation script
    image = image.resize(
        (256, 256),
        Image.BILINEAR
    )

    arr = np.asarray(
        image
    ).astype(
        np.float32
    ) / 255.0

    # Same OpenCV resize
    arr = cv2.resize(
        arr,
        (224, 224)
    )

    tensor = torch.from_numpy(
        arr
    ).permute(
        2, 0, 1
    ).float()

    tensor = (
        tensor -
        IMAGENET_MEAN
    ) / IMAGENET_STD

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# STANDARD PREDICTION
# ============================================================

@torch.inference_mode()
def predict_single(
    model,
    image_tensor
):

    image_tensor = image_tensor.to(
        DEVICE,
        non_blocking=True
    )

    logits = model(
        image_tensor
    )

    probabilities = F.softmax(
        logits,
        dim=1
    )

    return probabilities


# ============================================================
# 3-VIEW TTA
# EXACTLY MATCHES EVALUATION CODE
# ============================================================

@torch.inference_mode()
def predict_tta(
    model,
    image_tensor
):

    images = image_tensor.to(
        DEVICE,
        non_blocking=True
    )

    mean = (
        IMAGENET_MEAN
        .to(DEVICE)
        .unsqueeze(0)
    )

    std = (
        IMAGENET_STD
        .to(DEVICE)
        .unsqueeze(0)
    )

    # Convert normalized image back to pixels
    x_px = (
        images * std + mean
    ).clamp(0, 1)

    # View 1: original
    p1 = F.softmax(
        model(images),
        dim=1
    )

    # View 2: horizontal flip
    flipped = torch.flip(
        x_px,
        [3]
    )

    flipped = (
        flipped - mean
    ) / std

    p2 = F.softmax(
        model(flipped),
        dim=1
    )

    # View 3: center crop
    xc = F.interpolate(
        x_px,
        (256, 256),
        mode="bilinear",
        align_corners=False
    )[:, :, 16:240, 16:240]

    xc = (
        xc - mean
    ) / std

    p3 = F.softmax(
        model(xc),
        dim=1
    )

    final_probability = (
        p1 + p2 + p3
    ) / 3.0

    return (
        final_probability,
        p1,
        p2,
        p3
    )


# ============================================================
# FORMAT CLASS NAME
# ============================================================

def format_class_name(name):

    replacements = {
        "F15": "F-15",
        "F16": "F-16",
        "F18": "F/A-18",
        "F22": "F-22",
        "F35": "F-35",
        "F14": "F-14",
        "F4": "F-4",
        "FCK1": "FCK-1",
        "JF17": "JF-17",
        "J20": "J-20",
        "J35": "J-35",
        "J36": "J-36",
        "J10": "J-10",
        "Mig29": "MiG-29",
        "Mig31": "MiG-31",
        "Su24": "Su-24",
        "Su25": "Su-25",
        "Su34": "Su-34",
        "Su47": "Su-47",
        "Su57": "Su-57",
        "Rafale": "Dassault Rafale",
        "Mirage2000": "Mirage 2000",
        "EF2000": "Eurofighter Typhoon",
        "Tejas": "HAL Tejas",
        "KAAN": "KAAN",
    }

    return replacements.get(
        name,
        name
    )


# ============================================================
# TOP-K
# ============================================================

def get_top_predictions(
    probabilities,
    classes,
    k=5
):

    values, indices = torch.topk(
        probabilities,
        min(k, len(classes)),
        dim=1
    )

    values = values[0].detach().cpu().numpy()
    indices = indices[0].detach().cpu().numpy()

    results = []

    for probability, index in zip(
        values,
        indices
    ):

        results.append(
            {
                "class": classes[index],
                "display_name": format_class_name(
                    classes[index]
                ),
                "probability": float(
                    probability
                ),
                "index": int(index),
            }
        )

    return results


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("✈️ CUGAR ViT")

    st.markdown(
        "### Model"
    )

    st.write(
        "Causal–Uncertainty Guided "
        "Adaptive Refinement"
    )

    st.divider()

    st.subheader("Inference")

    use_tta = st.toggle(
        "3-View TTA",
        value=True,
        help=(
            "Original + horizontal flip + "
            "center crop. Matches evaluation."
        )
    )

    st.divider()

    st.subheader("System")

    st.write(
        f"**Device:** `{DEVICE}`"
    )

    if DEVICE.type == "cuda":

        st.write(
            f"**GPU:** `{torch.cuda.get_device_name(0)}`"
        )

        gpu_memory = torch.cuda.get_device_properties(
            0
        ).total_memory / (1024 ** 3)

        st.write(
            f"**VRAM:** `{gpu_memory:.1f} GB`"
        )

    else:

        st.warning(
            "CUDA unavailable. Running on CPU."
        )

    st.divider()

    st.caption(
        "Fine-Grained Fighter Aircraft "
        "Classification"
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with st.spinner("Loading CUGARViT model..."):

        model, class_names, checkpoint = (
            load_model()
        )

except Exception as e:

    st.error(
        "Failed to load model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '✈️ Fighter Aircraft Recognition'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Fine-grained military aircraft classification '
    'using CUGARViT'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">
                {len(class_names)}
            </div>
            <div class="metric-label">
                Aircraft Classes
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">
                ViT-B/16
            </div>
            <div class="metric-label">
                Backbone
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-value">
                224×224
            </div>
            <div class="metric-label">
                Input Size
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    tta_label = "Enabled" if use_tta else "Disabled"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">
                {tta_label}
            </div>
            <div class="metric-label">
                TTA
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")


# ============================================================
# IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an aircraft image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    help="Upload a clear image of a military aircraft."
)


if uploaded_file is None:

    st.info(
        "Upload an aircraft image to start classification."
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

except Exception as e:

    st.error(
        "Could not read the uploaded image."
    )

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE + PREDICTION
# ============================================================

image_col, prediction_col = st.columns(
    [1.05, 0.95],
    gap="large"
)


# ------------------------------------------------------------
# IMAGE
# ------------------------------------------------------------

with image_col:

    st.subheader("Input Image")

    st.image(
        image,
        use_container_width=True
    )

    st.caption(
        f"Original resolution: "
        f"{image.width} × {image.height}"
    )


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

with prediction_col:

    st.subheader("Classification")

    image_tensor = preprocess_image(
        image
    )

    # Warm up / synchronize timing
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    if use_tta:

        probabilities, p1, p2, p3 = (
            predict_tta(
                model,
                image_tensor
            )
        )

    else:

        probabilities = predict_single(
            model,
            image_tensor
        )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    inference_time = (
        time.perf_counter() -
        start_time
    )

    predictions = get_top_predictions(
        probabilities,
        class_names,
        NUM_TOP_PREDICTIONS
    )

    best = predictions[0]

    st.markdown(
        f"""
        <div class="prediction-box">

            <div style="font-size:14px;color:#777;">
                PREDICTED AIRCRAFT
            </div>

            <div class="prediction-name">
                {best["display_name"]}
            </div>

            <div class="confidence">
                Confidence:
                <b>{best["probability"] * 100:.2f}%</b>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.metric(
        "Inference Time",
        f"{inference_time * 1000:.1f} ms"
    )

    st.write("### Top Predictions")


    # --------------------------------------------------------
    # TOP 5 PROBABILITY BARS
    # --------------------------------------------------------

    for rank, prediction in enumerate(
        predictions,
        start=1
    ):

        label = (
            f"{rank}. "
            f"{prediction['display_name']}"
        )

        probability = (
            prediction["probability"]
        )

        st.write(
            f"**{label}** — "
            f"{probability * 100:.2f}%"
        )

        st.progress(
            probability
        )


# ============================================================
# TTA DETAILS
# ============================================================

if use_tta:

    st.divider()

    st.subheader(
        "3-View TTA Analysis"
    )

    tta_col1, tta_col2, tta_col3 = st.columns(3)

    view_predictions = [
        ("Original", p1),
        ("Horizontal Flip", p2),
        ("Center Crop", p3),
    ]

    for column, (
        view_name,
        view_probability
    ) in zip(
        [tta_col1, tta_col2, tta_col3],
        view_predictions
    ):

        view_top = get_top_predictions(
            view_probability,
            class_names,
            1
        )[0]

        with column:

            st.markdown(
                f"**{view_name}**"
            )

            st.write(
                view_top["display_name"]
            )

            st.write(
                f"{view_top['probability'] * 100:.2f}%"
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander(
    "Model Information"
):

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.write(
            "**Architecture:** CUGARViT"
        )

        st.write(
            "**Backbone:** Vision Transformer B/16"
        )

        st.write(
            "**Embedding Dimension:** 768"
        )

        st.write(
            "**Input:** 224 × 224 RGB"
        )

    with info_col2:

        st.write(
            f"**Classes:** {len(class_names)}"
        )

        st.write(
            "**Normalization:** ImageNet"
        )

        st.write(
            "**TAMRZ:** Enabled"
        )

        st.write(
            f"**Checkpoint:** "
            f"`{Path(CHECKPOINT_PATH).name}`"
        )


# ============================================================
# CHECKPOINT TRAINING INFORMATION
# ============================================================

with st.expander(
    "Checkpoint Information"
):

    if isinstance(checkpoint, dict):

        if "val_acc_single" in checkpoint:

            st.write(
                f"Validation Accuracy "
                f"(Single): "
                f"`{checkpoint['val_acc_single']}`"
            )

        if "val_acc_tta" in checkpoint:

            st.write(
                f"Validation Accuracy "
                f"(TTA): "
                f"`{checkpoint['val_acc_tta']}`"
            )

        if "classes" in checkpoint:

            st.write(
                f"Classes stored in checkpoint: "
                f"`{len(checkpoint['classes'])}`"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CUGARViT • Fine-Grained Fighter Aircraft Recognition"
)