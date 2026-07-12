import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disabilita GPU

import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib
from tensorflow.keras.models import load_model

# Configurazione Streamlit
st.set_page_config(page_title="PROSSIMITÀ 2026 - AI Simulator", layout="wide")

# 1. FUNZIONI DI CARICAMENTO ASSET
@st.cache_resource
def load_assets():
    model = load_model("modello_prossimita_v2.keras", compile=False)
    scaler = joblib.load("scaler_v2.gz")
    # Caricamento GeoJSON esagonale (deve essere in EPSG:4326)
    gdf = gpd.read_file("griglia_esagonale_output.geojson")
    return model, scaler, gdf

model, scaler, gdf_base = load_assets()

# 2. SCHEMA TAG DI DEFAULT
default_tags = {
    'Salute': {'Ospedale DEA': 1.0, 'Farmacia': 0.7, 'Parco Urbano': 0.5},
    'Istruzione': {'Università': 1.0, 'Scuola': 0.8, 'Biblioteca': 0.4},
    'Mobilità': {'Aeroporto': 1.0, 'Stazione Ferroviaria': 0.9, 'Fermata Metro/Bus': 0.6}
}

# --- SIDEBAR INTERATTIVA ---
st.sidebar.title("🎮 Laboratorio di Prossimità")
st.sidebar.markdown("Definisci i pesi delle risorse e aggiungi nuovi servizi.")

def get_pillar_attr(pillar_name):
    st.sidebar.subheader(f"📍 Pilastro {pillar_name}")
    
    # Pesi dei tag di default
    current_weights = []
    for tag, val in default_tags[pillar_name].items():
        if st.sidebar.checkbox(f"Attiva {tag}", value=True):
            w = st.sidebar.slider(f"Peso {tag}", 0.0, 1.0, val, key=f"sld_{tag}")
            current_weights.append(w)
    
    # TAG APERTI (3 Slot)
    st.sidebar.markdown(f"*Servizi Custom {pillar_name}*")
    for i in range(1, 4):
        c_name = st.sidebar.text_input(f"Nome Tag {i}", "", key=f"name_{pillar_name}_{i}")
        if c_name:
            c_weight = st.sidebar.slider(f"Peso {c_name}", 0.0, 1.0, 0.1, key=f"w_{pillar_name}_{i}")
            current_weights.append(c_weight)
            
    return max(current_weights) if current_weights else 0.0

# Calcolo Attrattività per i 3 Pilastri
attr_h = get_pillar_attr('Salute')
attr_e = get_pillar_attr('Istruzione')
attr_m = get_pillar_attr('Mobilità')

# Boost Infrastrutturale (PSI)
st.sidebar.markdown("---")
boost_psi = st.sidebar.slider("🚀 Boost Efficienza Reti (PSI)", 0.0, 0.4, 0.0)

# --- LOGICA DI CALCOLO AI ---
def run_simulation(gdf, a_h, a_e, a_m, b_psi):
    df = gdf.copy()
    # Boost PSI
    for c in ['psi_h', 'psi_e', 'psi_m']:
        df[c] = np.clip(df[c] + b_psi, 0, 1)
    
    # Assegnazione Attrattività
    df['attr_h'], df['attr_e'], df['attr_m'] = a_h, a_e, a_m
    
    # Features: [psi_h, psi_e, psi_m, attr_h, attr_e, attr_m, w_h, w_e, w_m]
    X = df[['psi_h', 'psi_e', 'psi_m', 'attr_h', 'attr_e', 'attr_m', 'w_h', 'w_e', 'w_m']].values
    X_scaled = scaler.transform(X)
    
    df['ip_ai'] = model.predict(X_scaled, verbose=0)
    return df

gdf_final = run_simulation(gdf_base, attr_h, attr_e, attr_m, boost_psi)

# --- VISUALIZZAZIONE ---
st.title("🧩 Mappa Dinamica della Prossimità Adattiva")
st.markdown(f"**Scenario:** Attrattività Salute: {attr_h:.2f} | Istruzione: {attr_e:.2f} | Mobilità: {attr_m:.2f}")

# Creazione Mappa
m = folium.Map(location=[41.893, 12.483], zoom_start=11, tiles='CartoDB Positron')

# Layer Esagonale
folium.Choropleth(
    geo_data=gdf_final,
    name="IP Dinamico",
    data=gdf_final,
    columns=["id_hex", "ip_ai"],
    key_on="feature.properties.id_hex",
    fill_color="Greys_r", # Toni di grigio richiesti per il libro
    fill_opacity=0.7,
    line_opacity=0.1,
    legend_name="Indice di Prossimità (AI)"
).add_to(m)

# Tooltip
folium.GeoJson(
    gdf_final,
    style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent'},
    tooltip=folium.GeoJsonTooltip(
        fields=['id_hex', 'ip_ai', 'w_h', 'w_e', 'w_m'],
        aliases=['Esagono:', 'IP AI:', 'Peso Salute:', 'Peso Edu:', 'Peso Mob:']
    )
).add_to(m)

st_folium(m, width=1200, height=650)

# Dashboard Statistica
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("IP Medio Area", f"{gdf_final['ip_ai'].mean():.3f}")
with c2:
    st.metric("Superficie Integrata (IP>0.7)", f"{(gdf_final['ip_ai'] > 0.7).sum()} es.")
with c3:
    st.metric("Marginalità (IP<0.3)", f"{(gdf_final['ip_ai'] < 0.3).sum()} es.")
