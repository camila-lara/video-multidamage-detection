# file: app.py
'''
This script creates a Streamlit app for real-time multiclass damage segmentation
using streamlit-webrtc and a trained BiSeNetV2 model. The app receives frames from
the browser camera, runs inference on each frame, and displays a multiclass overlay
blended over the live video stream. The model is preloaded before the WebRTC session
starts, and TURN configuration is optional.
'''

import os
import threading

import av
import cv2
import numpy as np
import streamlit as st
import torch
from streamlit_webrtc import webrtc_streamer, WebRtcMode

from bisenetv2_model import BiSeNetV2


# =========================================================
# CONFIG
# =========================================================
IMG_SIZE = 256
ALPHA_DEFAULT = 0.45
MODEL_PATH = "best_bisenetv2_multiclass.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)

CLASS_NAMES = ["background", "crack", "spalling", "corrosion"]
PALETTE_BGR = np.array([
    [0,   0,   0],    # 0 background
    [255, 255, 255],  # 1 crack
    [0,   0, 255],    # 2 spalling
    [0, 255, 255],    # 3 corrosion
], dtype=np.uint8)

MODEL = None


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "model_state" in ckpt_obj:
            return ckpt_obj["model_state"]
        if "state_dict" in ckpt_obj:
            return ckpt_obj["state_dict"]
    return ckpt_obj


def extract_logits(model_output):
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    model = BiSeNetV2(n_classes=4, aux_heads=True).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = extract_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def preprocess(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    x = resized.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x).float().to(DEVICE)
    return x


def predict_class_map(model, frame_bgr):
    h, w = frame_bgr.shape[:2]
    x = preprocess(frame_bgr)

    with torch.no_grad():
        output = model(x)
        logits = extract_logits(output)
        pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    return pred


def colorize_mask(class_map):
    return PALETTE_BGR[class_map]


def make_overlay(frame_bgr, class_map, alpha):
    color_mask = colorize_mask(class_map)
    blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, color_mask, alpha, 0.0)
    return blended


def get_ice_servers():
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]

    if (
        "TURN_URL" in st.secrets
        and "TURN_USERNAME" in st.secrets
        and "TURN_CREDENTIAL" in st.secrets
    ):
        ice_servers.append(
            {
                "urls": [st.secrets["TURN_URL"]],
                "username": st.secrets["TURN_USERNAME"],
                "credential": st.secrets["TURN_CREDENTIAL"],
            }
        )

    return ice_servers


# =========================================================
# VIDEO PROCESSOR
# =========================================================
class VideoProcessor:
    def __init__(self):
        global MODEL

        if MODEL is None:
            MODEL = load_model()

        self.model = MODEL
        self.lock = threading.Lock()
        self.alpha = ALPHA_DEFAULT

    # def recv(self, frame):
    #     img = frame.to_ndarray(format="bgr24")

    #     with self.lock:
    #         alpha = self.alpha

    #     class_map = predict_class_map(self.model, img)
    #     out = make_overlay(img, class_map, alpha)

    #     return av.VideoFrame.from_ndarray(out, format="bgr24")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# =========================================================
# APP
# =========================================================
def main():
    global MODEL

    st.set_page_config(page_title="Daño estructural en tiempo real", layout="wide")
    st.title("Segmentación multiclase de daño estructural en tiempo real")

    st.write(
        "Pulsa Start, permite acceso a la cámara del navegador y la app mostrará "
        "el overlay de daño estructural sobre el video en tiempo real."
    )

    st.info(
        f"Dispositivo usado por el modelo: {DEVICE}. "
        "En Streamlit Cloud normalmente correrá en CPU."
    )

    with st.sidebar:
        st.header("Parámetros")
        alpha = st.slider("Alpha overlay", 0.05, 0.95, ALPHA_DEFAULT, 0.05)

    with st.spinner("Cargando modelo..."):
        MODEL = load_model()

    ice_servers = get_ice_servers()

    if len(ice_servers) == 1:
        st.caption("WebRTC configurado con STUN. Si la cámara no conecta en algunas redes, puede ser necesario agregar TURN.")
    else:
        st.caption("WebRTC configurado con STUN + TURN.")

    webrtc_streamer(
        key="damage-realtime-multiclass",
        mode=WebRtcMode.SENDRECV,
        frontend_rtc_configuration={"iceServers": ice_servers},
        server_rtc_configuration={"iceServers": ice_servers},
        media_stream_constraints={
            "video": {"facingMode": {"ideal": "environment"}},
            "audio": False,
        },
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

    processor_state = st.session_state.get("damage-realtime-multiclass")
    if processor_state and hasattr(processor_state, "video_processor") and processor_state.video_processor:
        with processor_state.video_processor.lock:
            processor_state.video_processor.alpha = alpha

    st.markdown(
        """
        Uso:
        1. Pulsa **Start**.
        2. Acepta permisos de cámara en el navegador.
        3. Ajusta el alpha si lo necesitas.
        4. En teléfono o tablet, el navegador puede intentar usar la cámara trasera.
        """
    )

    st.subheader("Clases:")
    st.markdown(
        """
        **Grieta:** Blanco  
        **Pérdida de recubrimiento:** Rojo  
        **Corrosión:** Amarillo
        """
    )


if __name__ == "__main__":
    main()