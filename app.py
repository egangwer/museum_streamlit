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


st.set_page_config(layout="wide")
page = st.sidebar.radio('Select a page', ['Recommendations', 'Museum Data'])

museum_dat = load("data/FINAL_museum_dat_with_extra.joblib")
kmeans_model = load("data/FINAL_kmeans.joblib")

def home_page():
    st.title('Museum Artwork Topic Modeling')

    def recommend_artworks(title, df, kmeans_model):
        artwork = df[df['Title'].str.lower() == title.lower()]
        if artwork.empty:
            return "Artwork not found! Try again!"
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
                #    st.caption(f"{row['Description']}")
                else: 
                    st.warning("Image not found. Refresh for new picks!")
                    st.write(f"**{row['Title']}** by **{row['Artist']}** is made of {row['Medium']}. This artwork is currentlyin the **{row['Museum']} Museum Collection**.")
                #    st.caption(f"'{row['Description']}'")
    else: 
        user_input = st.text_input('Enter an artist here:')
        if st.button("Find artist"):
            results = find_artist(user_input, museum_dat)
            if isinstance(results, str):
                st.warning(results)
            else:
                st.subheader("Artist Information:")
                st.dataframe(results)


    st.subheader("Looking for inspiration? Check out these artworks!")
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

    st.write("Data is from the Cleveland Art Museum, Museum of Modern Art in New York, Art Institute of Chicago, and the National Gallery of Art artwork collections. Data was accessed through API's and web scraping.")
    st.write('The bertopic model was trained on artwork description data and then futher clustered using KMeans.')


if page == 'Recommendations':
    home_page()

elif page == 'Museum Data':
    import page2  
    page2.show()