# file: app.py
'''
This script creates a Streamlit app for real-time multiclass damage segmentation
using streamlit-webrtc and a trained BiSeNetV2 model. The app receives frames from
the browser camera, runs inference on each frame, and displays only the multiclass
overlay blended over the live video stream.
'''

import os
import threading
import av
import cv2
import numpy as np
import streamlit as st
import torch
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from twilio.rest import Client
from bisenetv2_model import BiSeNetV2

# =========================================================
# CONFIGURACIÓN
# =========================================================
IMG_SIZE = 256
ALPHA_DEFAULT = 0.45
MODEL_PATH = "best_bisenetv2_multiclass.pth"

# Forzar CPU si no hay GPU disponible (como en Streamlit Cloud)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)

PALETTE_BGR = np.array([
    [0,   0,   0],    # 0 background
    [255, 255, 255],  # 1 crack
    [0,   0, 255],    # 2 spalling
    [0, 255, 255],    # 3 corrosion
], dtype=np.uint8)

# --- Función para obtener configuración STUN/TURN de Twilio ---
@st.cache_data
def get_ice_servers():
    """
    Intenta obtener servidores TURN de Twilio para evitar errores de conexión (WebRTC).
    Si no hay secretos configurados, cae de nuevo al STUN de Google.
    """
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        client = Client(account_sid, auth_token)
        token = client.tokens.create()
        return token.ice_servers
    except (KeyError, Exception):
        st.warning("No se encontraron credenciales de Twilio. Usando servidor STUN público (puede fallar en redes móviles).")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]


# --- Carga del modelo optimizada ---
@st.cache_resource
def load_model():
    model = BiSeNetV2(n_classes=4)
    
    # 1. Cargamos el archivo .pth completo
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # 2. Extraemos ÚNICAMENTE los pesos del modelo (ignorando epoch y métricas)
    # Verificamos si es un diccionario que contiene la llave 'model_state'
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint # Por si acaso los pesos vinieran directos
        
    # 3. Ahora sí, cargamos solo los pesos al modelo
    model.load_state_dict(state_dict)
    
    model.to(DEVICE)
    model.eval()
    return model

# --- Procesador de Video ---
class VideoProcessor:
    def __init__(self):
        self.model = load_model()
        self.alpha = ALPHA_DEFAULT
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Preprocesamiento
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_norm = (img_resized.astype(np.float32) / 255.0 - MEAN) / STD
        img_input = img_norm.transpose(2, 0, 1)[np.newaxis, ...]
        tensor_input = torch.from_numpy(img_input).to(DEVICE)

        # Inferencia
        with torch.no_grad():
            output = self.model(tensor_input)
            if isinstance(output, (list, tuple)):
                output = output[0]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Postprocesamiento (Color y Overlay)
        mask_color = PALETTE_BGR[pred]
        mask_color = cv2.resize(mask_color, (w, h), interpolation=cv2.INTER_NEAREST)

        with self.lock:
            alpha = self.alpha

        # Mezclar imagen original con la máscara detectada
        combined = cv2.addWeighted(img, 1 - alpha, mask_color, alpha, 0)

        return av.VideoFrame.from_ndarray(combined, format="bgr24")

# =========================================================
# INTERFAZ STREAMLIT
# =========================================================
def main():
    global MODEL
    st.set_page_config(page_title="Daño estructural en tiempo real", layout="wide")
    st.title("Segmentación multiclase de daño estructural en tiempo real")

    st.write(
        "Pulsa Start, permite acceso a la cámara y la app mostrará "
        "el overlay multiclase sobre el video en tiempo real."
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

    # Configuración de red para el streaming
    rtc_configuration = RTCConfiguration({"iceServers": get_ice_servers()})

    webrtc_ctx = webrtc_streamer(
        key="damage-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": {"facingMode": "environment"}, # "environment" intenta usar la cámara trasera
            "audio": False,
        },
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        with webrtc_ctx.video_processor.lock:
            webrtc_ctx.video_processor.alpha = alpha

    st.markdown("""
    **Instrucciones:**
    1. Haz clic en **Start** para iniciar la cámara.
    2. Si estás en móvil, se intentará usar la cámara trasera por defecto.
    3. Ajusta la opacidad desde la barra lateral.
    """)

if __name__ == "__main__":
    main()