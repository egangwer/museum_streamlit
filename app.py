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


st.set_page_config(layout="wide")

museum_dat = load("data/FINAL_museum_dat_with_extra.joblib")
kmeans_model = load("data/FINAL_kmeans.joblib")

st.title('Museum Artwork Topic Modeling')

def recommend_artworks(title, df, kmeans_model):
    artwork = df[df['Title'].str.lower() == title.lower()]
    if artwork.empty:
        return "Artwork not found. Try again."
    cluster_label = artwork['Cluster'].values[0] 
    similar_artworks = df[df['Cluster'] == cluster_label] 
    # similar_artworks['Artist'] = similar_artworks['Artist'].str.title()
    return similar_artworks[['Title', 'Artist', 'Date_Start', 'Date_End', 'Medium', 'Image_Link', 'Museum', 'Description']].sample(1)


def find_artist(artist, df):
    artist = df[df['Artist_Clean'].str.lower() == artist.lower()]
    if artist.empty:
        return "Artist not found. Try again."
    artist_df = df[df['Artist_Clean'] == artist['Artist_Clean'].values[0]]
    artist_df['Artist_Clean'] = artist_df['Artist_Clean'].str.title()
    return artist_df[['Artist_Clean', 'Title', 'Date_Start', 'Date_End', 'Medium', 'Image_Link']].drop_duplicates()


search_type = st.selectbox("I want to...", ["Find Similar Artworks", "Find an Artist"])

if search_type == "Find Similar Artworks":
    user_input = st.text_input('Enter an artwork here:')
    if st.button("Recommend similar artworks"):
        results = recommend_artworks(user_input, museum_dat, kmeans_model)
        if isinstance(results, str):
            st.warning(results)
        else:
            st.subheader(f"Artwork similiar to '{user_input}':")
            row = results.iloc[0]
            if pd.notna(row["Image_Link"]):
                if row["Image_Link"]:
                    st.image(row["Image_Link"], width=300)
                else:
                    st.write("Image not available")
                st.markdown(f" **{row['Title']}** by **{row['Artist']}** is made of {row['Medium']}. This artwork is currently in the **{row['Museum']} Museum Collection**.")
                st.caption(f"{row['Description']}")
            else: 
                st.warning("Image not found. Refresh for new picks!")
                st.write(f"**{row['Title']}** by **{row['Artist']}** is made of {row['Medium']}. This artwork is currentlyin the **{row['Museum']} Museum Collection**.")
                st.caption(f"'{row['Description']}'")
else: 
    user_input = st.text_input('Enter an artist here:')
    if st.button("Find artist"):
        results = find_artist(user_input, museum_dat)
        if isinstance(results, str):
            st.warning(results)
        else:
            st.subheader("Artist Information:")
            st.dataframe(results)


st.subheader("Looking for inspiration? Start with these random artworks!")
random_artworks = museum_dat.sample(4, random_state=random.randint(1, 23000))  
cols = st.columns(4)

for idx, (col, row) in enumerate(zip(cols, random_artworks.iterrows())):
    with col:
        image_url = row[1].get("Image_Link", "")  # Get the image link safely
        if isinstance(image_url, str) and image_url.strip():  # Check if it's a valid string
            st.image(image_url, width=120)
            st.caption(f"**{row[1]['Title']}**\n_{row[1]['Artist']}, {row[1]['Date_Start']} - {row[1]['Date_End']}_")
        else:
            st.warning("Image not found. Refresh for new picks!")
            st.caption(f"**{row[1]['Title']}**\n_{row[1]['Artist']}, {row[1]['Date_Start']} - {row[1]['Date_End']}_")
       
        

if st.button("Refresh Artworks"):
    st.rerun()

st.write("Data is from the Cleveland Art Museum, Museum of Modern Art in New York, Art Institute of Chicago, and the National Gallery of Art artwork collections.")
st.write("The data was accessed through API's and web scraping.")
st.write('The bertopic model was trained on artwork description data and then futher clustered using KMeans.')

### Data Visualizations  
st.subheader('Artwork Dataset Stats')
steps = 50
delay = 0.05  # Speed of animation
stats = {
    "Total Artworks": (23586, "#00C0C1", 50),
    "Museums Featured": (4, "#008F94", 50),
    "Total Artists": (5570, "#008F94", 50),
    "Mediums": (300, "#008F94", 50) 
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

fig.update_traces(marker_color= "#008F94", textposition="inside",
                  textfont=dict(size=10, family="Arial Black", color="white"))
fig.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig)

# Artwork Accession Year by museum 
museum_colors = {
    "Metropolitan Museum of Art": "#008F94",
    "Art Institute of Chicago": "#00C0C1",
    "Cleveland Museum of Art": "#00709E",
    "National Gallery of Art": "#013753"
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
