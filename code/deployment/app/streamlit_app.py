"""
Stage 3: Web application
Simple UI that collects match parameters and shows the API prediction.
"""

import os

import requests
import streamlit as st

# Inside docker-compose the API is reachable by its service name
API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(page_title="ATP Match Predictor", page_icon="🎾")
st.title("🎾 ATP Match Winner Prediction")
st.write("Enter the match parameters and get a prediction of the winner.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Player 1")
    rank_1 = st.number_input("ATP rank", min_value=1, max_value=500, value=10, key="r1")
    pts_1 = st.number_input("ATP points", min_value=0, value=3000, key="p1")

with col2:
    st.subheader("Player 2")
    rank_2 = st.number_input("ATP rank", min_value=1, max_value=500, value=50, key="r2")
    pts_2 = st.number_input("ATP points", min_value=0, value=900, key="p2")

st.subheader("Match conditions")

series = st.selectbox("Series", ["Grand Slam", "Masters 1000", "ATP500", "ATP250", "Masters Cup"])
court = st.selectbox("Court", ["Outdoor", "Indoor"])
surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
round_name = st.selectbox(
    "Round",
    ["1st Round", "2nd Round", "3rd Round", "4th Round",
     "Quarterfinals", "Semifinals", "The Final", "Round Robin"],
)
best_of = st.radio("Best of", [3, 5], horizontal=True)

if st.button("Predict winner", type="primary"):
    payload = {
        "rank_1": rank_1,
        "rank_2": rank_2,
        "pts_1": pts_1,
        "pts_2": pts_2,
        "series": series,
        "court": court,
        "surface": surface,
        "round_name": round_name,
        "best_of": best_of,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        st.success(f"Predicted winner: **{result['winner']}**")
        st.metric("Model confidence", f"{result['confidence'] * 100:.1f}%")

    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the API: {e}")