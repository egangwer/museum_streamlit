import streamlit as st
import pandas as pd
import random
import nltk
from joblib import load, dump
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import time
from importlib import import_module



def show(): 
    museum_dat = load("data/FINAL_museum_dat_with_extra.joblib")
    kmeans_model = load("data/FINAL_kmeans.joblib")

    ### Data Visualizations  
    st.title('Artwork Dataset Stats')
    steps = 50
    delay = 0.05  # Speed of animation
    stats = {
        "Total Artworks": (23586, "#DAE050", 50),
        "Museums Featured": (4, "#1E4DD9", 50),
        "Total Artists": (5570, "#1E4DD9", 50),
        "Mediums": (300, "#1E4DD9", 50) 
    }
    cols = st.columns(len(stats))
    placeholders = {key: col.empty() for key, col in zip(stats.keys(), cols)}

    css_pulse_fade_glow = """
    <style>
    @keyframes pulse-fade-glow {
        0% { opacity: 0; transform: scale(0.9); text-shadow: none; }
        50% { opacity: 1; transform: scale(1.1); text-shadow: 0px 0px 20px rgba(255, 255, 255, 0.8); }
        100% { opacity: 1; transform: scale(1); text-shadow: none; }
    }
    .pulsate-fade-glow {
        animation: pulse-fade-glow 1.5s ease-in-out;
    }
    </style>
    """
    st.markdown(css_pulse_fade_glow, unsafe_allow_html=True)  # Inject CSS


    # Set the maximum number of steps (used for slower stats)
    max_steps = max(speed for _, _, speed in stats.values())

    # Animate the numbers
    for i in range(max_steps + 1):
        for key, (target, color, speed) in stats.items():
            if i % (max_steps // speed) == 0:  # Controls speed per stat
                value = int((i / speed) * target) if i < speed else target  # Gradual increase, stops at target
                
                # Apply fade-in + pulse class ONLY at the final value
                pulse_class = "pulsate-fade-glow" if value == target else ""

                # Custom HTML for each BAN
                styled_text = f"""
                <div class="{pulse_class}" style="font-size: 50px; color: {color}; font-weight: bold; text-align: center;">
                    {value:,}
                </div>
                <p style="font-size: 18px; text-align: center; color: #555;">{key}</p>
                """
                
                placeholders[key].markdown(styled_text, unsafe_allow_html=True)
        
        time.sleep(0.05) 



    st.subheader("Artwork Count by Museum")
    museum_counts = museum_dat["Museum"].value_counts().reset_index()
    museum_counts.columns = ["Museum", "Artwork Count"]
    museum_counts["Formatted Count"] = museum_counts["Artwork Count"].apply(lambda x: f"{x:,}")

    fig = px.bar(museum_counts, x="Artwork Count", y="Museum",text="Formatted Count", orientation="h")

    fig.update_traces(marker_color= "#1E4DD9", textposition="inside",
                    textfont=dict(size=10, family="Arial Black", color="white"))
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig)

    # Artwork Accession Year by museum 
    museum_colors = {
        "Metropolitan Museum of Art": "#1E4DD9",
        "Art Institute of Chicago": "#555FDB",
        "Cleveland Museum of Art": "#D1DB68",
        "National Gallery of Art": "#3E5084"
    }
    st.subheader("Artwork Accession Year by Museum")

    @st.cache_data
    def load_data(museum_dat=museum_dat):
        museum_dat = museum_dat.dropna(subset=["AccessionYear"])
        museum_yearly_counts = museum_dat.groupby(["AccessionYear", "Museum"]).size().reset_index(name="Count")
        museum_yearly_counts["AccessionYear"] = museum_yearly_counts["AccessionYear"].astype(str)
        return museum_yearly_counts

    museum_yearly_counts = load_data()

    museum_yearly_counts["Smoothed Count"] = museum_yearly_counts.groupby("Museum")["Count"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    museum_yearly_counts = museum_yearly_counts.sort_values(by=["AccessionYear", "Museum"])

    chart_placeholder = st.empty()
    years = sorted(museum_yearly_counts["AccessionYear"].unique())
    years = years[::5]

    for year in years:
        filtered_data = museum_yearly_counts[museum_yearly_counts["AccessionYear"] <= year]
        fig = px.line(filtered_data, x="AccessionYear", y="Smoothed Count", color="Museum", markers=True, color_discrete_map=museum_colors)
        fig.update_traces(line=dict(width=2), marker=dict(size=2))
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.04)