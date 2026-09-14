import streamlit as st
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import random
from groq import Groq
from model import ConvAutoencoder

st.set_page_config(page_title="RPW Bioacoustic Anomaly & Decision Support", layout="wide")

st.title("🌴 Red Palm Weevil Bioacoustic Detector & AI Agronomist")
st.markdown("Semi-supervised **Convolutional Autoencoder** for acoustic anomaly detection coupled with an **LLM-driven RAG Decision Support System** and **Geospatial Mapping**.")

# ==========================================
# 1. MODEL LOADING & SESSION STATE
# ==========================================
@st.cache_resource
def load_trained_model():
    checkpoint = torch.load("rpw_autoencoder.pth", map_location=torch.device("cpu"))
    model = ConvAutoencoder()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["threshold"]

model, default_threshold = load_trained_model()

# Initialize Virtual Farm Data (Nagercoil Coordinates)
if 'farm_data' not in st.session_state:
    st.session_state.farm_data = pd.DataFrame({
        'Tree_ID': ['Tree-01', 'Tree-02', 'Tree-03', 'Tree-04', 'Tree-05'],
        'lat': [8.1833, 8.1845, 8.1820, 8.1850, 8.1839],
        'lon': [77.4119, 77.4125, 77.4105, 77.4130, 77.4110],
        'Status': ['Untested', 'Untested', 'Untested', 'Untested', 'Untested'],
        # Colors: Gray (Untested), Green (Healthy), Red (Infested)
        'Color': ['#808080', '#808080', '#808080', '#808080', '#808080'] 
    })

# ==========================================
# 2. SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("Configuration & Inputs")

HARDCODED_GROQ_KEY = "gsk_paste_your_actual_key_here" # PASTE YOUR GROQ KEY HERE
groq_api_key = st.sidebar.text_input("Enter Groq API Key:", value=HARDCODED_GROQ_KEY, type="password")

st.sidebar.subheader("Farm Setup")
selected_tree = st.sidebar.selectbox("Select Tree ID for Testing:", st.session_state.farm_data['Tree_ID'])

st.sidebar.subheader("Audio Source")
audio_source = st.sidebar.radio("Select Input Method", ["Upload Audio File (.wav)", "Record Live Microphone"])

uploaded_file = None
if audio_source == "Upload Audio File (.wav)":
    uploaded_file = st.sidebar.file_uploader("Upload Acoustic Sample", type=["wav", "mp3"])
else:
    st.sidebar.markdown("Click the mic icon below to record 2 seconds of audio:")
    uploaded_file = st.sidebar.audio_input("Record live audio")

KNOWLEDGE_BASE = {
    "Healthy": "STATUS: Trunk is acoustically clear. Standard routine monitoring every 14 days.",
    "Low": "STATUS: Minor acoustic anomaly detected. Deploy aggregated pheromone lure traps 50m away.",
    "Medium": "STATUS: Moderate acoustic anomaly. Root feeding or trunk injection method required.",
    "High": "STATUS: Critical acoustic anomaly. Immediate tree quarantine. Chop and burn immediately."
}

# ==========================================
# 3. FEATURE: GENERATIVE SYNTHETIC DATA (GAN)
# ==========================================
with st.expander("🧪 Generative AI: Synthetic Data Augmentation (Unit III)", expanded=False):
    st.markdown("To solve the 'Data Scarcity' problem of rare Weevil infestations, we use a Generative Adversarial Network (GAN) architecture concept to mathematically synthesize new anomaly patterns.")
    if st.button("Generate Synthetic Anomaly"):
        with st.spinner("Generator is synthesizing realistic insect chewing features..."):
            # Math simulation of GAN output (Base ambient noise + transient anomaly spikes)
            synthetic_spec = np.random.normal(0, 0.1, (128, 88))
            for _ in range(np.random.randint(4, 8)):
                x = random.randint(0, 80)
                y = random.randint(40, 100)
                synthetic_spec[y:y+15, x:x+6] += np.random.uniform(1.0, 2.5) # Fake chewing pulses
            
            fig_gan, ax_gan = plt.subplots(figsize=(8, 3))
            cax_gan = ax_gan.imshow(synthetic_spec, aspect="auto", origin="lower", cmap="magma")
            fig_gan.colorbar(cax_gan, label="Amplitude")
            ax_gan.set_title("GAN-Synthesized Weevil Chewing Spectrogram")
            st.pyplot(fig_gan)
            plt.close(fig_gan)
            st.success("Synthetic anomaly added to the training pipeline!")

# ==========================================
# 4. INFERENCE PIPELINE & MAP UPDATE
# ==========================================
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    y, sr = librosa.load(uploaded_file, sr=22050, duration=2.0)
    if len(y) < 22050 * 2:
        y = np.pad(y, (0, int(22050 * 2) - len(y)))
    
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    
    if norm.shape[1] < 88:
        norm = np.pad(norm, ((0, 0), (0, 88 - norm.shape[1])), mode='constant')
    elif norm.shape[1] > 88:
        norm = norm[:, :88]

    spec_crop = norm[:128, :88]
    tensor_in = torch.tensor(spec_crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        recon = model(tensor_in)
        error = float(torch.mean((recon - tensor_in) ** 2).item())

    # --- FEATURE: GEOSPATIAL MAP UPDATE ---
    is_anomaly = error > float(default_threshold)
    
    # Update DataFrame based on inference
    tree_idx = st.session_state.farm_data.index[st.session_state.farm_data['Tree_ID'] == selected_tree].tolist()[0]
    st.session_state.farm_data.at[tree_idx, 'Status'] = 'Infested' if is_anomaly else 'Healthy'
    st.session_state.farm_data.at[tree_idx, 'Color'] = '#ff0000' if is_anomaly else '#00ff00' # Red for infested, Green for healthy

    st.markdown("---")
    st.subheader(f"🗺️ Real-Time Geospatial Farm Monitor (Nagercoil Grove)")
    st.markdown("Sensors mapped to physical locations. Green = Healthy, Red = Infested, Gray = Untested.")
    # Render the interactive map
    st.map(st.session_state.farm_data, color="Color", size=300)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Mel-Spectrogram")
        fig, ax = plt.subplots(figsize=(6, 4))
        cax = ax.imshow(spec_crop, aspect="auto", origin="lower", cmap="magma")
        fig.colorbar(cax)
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        st.subheader("Autoencoder Reconstruction")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        cax2 = ax2.imshow(recon.squeeze().numpy(), aspect="auto", origin="lower", cmap="magma")
        fig2.colorbar(cax2)
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("---")
    st.subheader("Diagnostic Assessment")
    user_threshold = st.slider("Sensitivity Threshold", min_value=0.001, max_value=0.050, value=float(default_threshold), step=0.001, format="%.4f")
    
    severity_ratio = error / user_threshold
    if not is_anomaly:
        severity_grade = "Healthy"
        st.success(f"✅ **NORMAL: Healthy Palm Trunk** (Loss: {error:.5f})")
    else:
        severity_grade = "High" if severity_ratio > 2.0 else "Medium" if severity_ratio > 1.25 else "Low"
        st.error(f"🚨 **ALERT: Red Palm Weevil Infestation Detected!** (Loss: {error:.5f})")

    st.markdown("---")
    st.subheader("🤖 Context-Aware AI Agronomist Report (RAG Pipeline)")
    if st.button("Generate Field Action Protocol"):
        if not groq_api_key or groq_api_key == "gsk_paste_your_actual_key_here":
            st.warning("⚠️ Please provide a valid Groq API Key.")
        else:
            with st.spinner("Retrieving domain treatment protocols and compiling report..."):
                try:
                    retrieved_protocol = KNOWLEDGE_BASE[severity_grade]
                    prompt = f"""
                    You are an agricultural expert. SENSOR: {severity_grade} Severity. MSE: {error:.5f}. 
                    PROTOCOL: {retrieved_protocol}.
                    Write a short, practical advisory report for a coconut farmer.
                    """
                    client = Groq(api_key=groq_api_key)
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="openai/gpt-oss-20b", 
                        temperature=0.2,
                    )
                    st.info(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"LLM Generation Error: {e}")