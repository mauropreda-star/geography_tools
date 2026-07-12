codePython
import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

st.set_page_config(page_title="Simulatore Prossimità Avanzato", layout="wide")

@st.cache_resource
def load_assets():
    model = load_model("modello_prossimita_v2.keras", compile=False)
    scaler = joblib.load("scaler_v2.gz")
    return model, scaler

model, scaler = load_assets()

st.title("🏙️ Urban Intelligence: Simulatore di Prossimità Qualitativa")
st.markdown("Analisi della cittadinanza basata su Distanza, Qualità delle Risorse e Bisogno Sociale.")

# --- SIDEBAR: LE TRE MANOPOLE DEL TERRITORIO ---
st.sidebar.header("⚙️ Parametri di Simulazione")

# 1. EFFICIENZA (PSI)
st.sidebar.subheader("1. Efficienza (Distanza/Routing)")
psi_h = st.sidebar.slider("Accesso Salute", 0.0, 1.0, 0.5)
psi_e = st.sidebar.slider("Accesso Istruzione", 0.0, 1.0, 0.5)
psi_m = st.sidebar.slider("Accesso Mobilità", 0.0, 1.0, 0.5)

# 2. QUALITÀ RISORSE (ATTR)
st.sidebar.subheader("2. Qualità/Importanza Risorse")
att_h = st.sidebar.slider("Rilevanza Presidi Sanitari", 0.0, 1.0, 0.8)
att_e = st.sidebar.slider("Rilevanza Scuole/Uni", 0.0, 1.0, 0.8)
att_m = st.sidebar.slider("Rilevanza Hub Trasporto", 0.0, 1.0, 0.8)

# 3. BISOGNO (PESI DEMOGRAFICI)
st.sidebar.subheader("3. Domanda Sociale (Pesi)")
w_h = st.sidebar.slider("Peso Salute (Anziani)", 0.0, 1.0, 0.33)
w_e = st.sidebar.slider("Peso Istruzione (Giovani)", 0.0, 1.0, 0.33)
w_m = st.sidebar.slider("Peso Mobilità (Pendolari)", 0.0, 1.0, 0.34)

# Normalizzazione pesi demografici
total_w = w_h + w_e + w_m
wh_n, we_n, wm_n = w_h/total_w, w_e/total_w, w_m/total_w

# --- CALCOLO PREDIZIONE ---
input_vec = np.array([[psi_h, psi_e, psi_m, att_h, att_e, att_m, wh_n, we_n, wm_n]])
input_scaled = scaler.transform(input_vec)
ip_pred = model.predict(input_scaled, verbose=0)[0][0]

# --- VISUALIZZAZIONE ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("Indice di Prossimità Dinamico")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = ip_pred,
        gauge = {'axis': {'range': [0, 1]}, 'bar': {'color': "black"},
                 'steps' : [{'range': [0, 0.4], 'color': "red"}, {'range': [0.4, 0.7], 'color': "orange"}, {'range': [0.7, 1], 'color': "green"}]}
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

with c2:
    st.subheader("Bilanciamento del Territorio")
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[psi_h * att_h, psi_e * att_e, psi_m * att_m], # Valore effettivo: Distanza * Qualità
        theta=['Salute','Istruzione','Mobilità'],
        fill='toself', line_color='blue', name='Valore Offerta'
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(fig_radar, use_container_width=True)

# --- ANALISI DEL RISULTATO ---
st.info(f"**Analisi AI:** Con una domanda sociale focalizzata al {wh_n*100:.1f}% sulla salute, "
        f"la qualità attuale degli ospedali (pari a {att_h}) determina un indice di cittadinanza { 'Sufficiente' if ip_pred > 0.5 else 'Critico'}.")
