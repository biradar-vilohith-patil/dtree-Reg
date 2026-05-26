import streamlit as st
import plotly.express as px
import pandas as pd
from src.predict import run_inference

st.set_page_config(page_title="AI Hit Predictor", page_icon="🎧", layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("🎧 Track Hit Predictor (AI)")
st.markdown("Design your track's audio profile to predict its Spotify popularity score.")
st.markdown("---")

col_inputs, col_visuals = st.columns([1, 1.2], gap="large")

with col_inputs:
    st.markdown("### 🎛️ Audio Mixing Board")
    
    track_genre = st.selectbox("Track Genre", ["pop", "hip-hop", "edm", "rock", "indie", "r-n-b", "acoustic"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    danceability = st.slider("Danceability", 0.0, 1.0, 0.75, 0.01)
    energy = st.slider("Energy", 0.0, 1.0, 0.82, 0.01)
    valence = st.slider("Valence (Happiness)", 0.0, 1.0, 0.60, 0.01)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.10, 0.01)
    tempo = st.slider("Tempo (BPM)", 60, 200, 128)

with col_visuals:
    st.markdown("### 📊 Audio Fingerprint")
    
    df_radar = pd.DataFrame(dict(
        r=[danceability, energy, valence, acousticness, tempo/200], 
        theta=['Danceability', 'Energy', 'Valence', 'Acousticness', 'Tempo']
    ))
    
    fig = px.line_polar(df_radar, r='r', theta='theta', line_close=True, range_r=[0, 1])
    fig.update_traces(fill='toself', fillcolor='rgba(29, 185, 84, 0.4)', line_color='#1db954', line_width=2)
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=20, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr style='border-color: #282828;'>", unsafe_allow_html=True)

if st.button("Predict Popularity Score 🚀"):
    user_data = {
        'danceability': danceability,
        'energy': energy,
        'valence': valence,
        'acousticness': acousticness,
        'tempo': tempo,
        'track_genre': track_genre
    }
    
    score = run_inference(user_data)
    
    if score >= 75:
        st.success(f"### 🔥 Viral Hit Detected!\n# Predicted Score: **{score:.1f} / 100**")
    elif score >= 50:
        st.info(f"### 📈 Solid Performer\n# Predicted Score: **{score:.1f} / 100**")
    else:
        st.warning(f"### 🎷 Niche / Underground\n# Predicted Score: **{score:.1f} / 100**")