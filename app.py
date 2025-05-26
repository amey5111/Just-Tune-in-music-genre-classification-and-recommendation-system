# app.py

import streamlit as st
import os
import numpy as np
import tensorflow as tf
import pickle
import requests
from utils.audio_processing import preprocess_audio

# Load model
model = tf.keras.models.load_model('genre_cnn_model.h5')

# Load label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Jamendo API details
JAMENDO_CLIENT_ID = st.secrets["JAMENDO_CLIENT_ID"]  

# Title
st.title("🎵 Just Tune.in")

st.subheader("Genre Classification & Music Recommendation System")
# Step 1: List 10 GTZAN songs
song_dir = 'gtzan_songs'
song_files = [f for f in os.listdir(song_dir) if f.endswith('.wav')]

st.header("🎶 Select a song to listen")

selected_song = st.selectbox("Pick a song:", song_files)

# Audio player
st.audio(os.path.join(song_dir, selected_song))

# Button to trigger prediction
if st.button("🎧 Predict Genre & Recommend Songs"):

    st.write("🔍 Processing...")

    # Step 2: Process audio
    audio_path = os.path.join(song_dir, selected_song)
    mel = preprocess_audio(audio_path)

    # Step 3: Predict
    mel_input = np.expand_dims(mel, axis=0)  # batch dimension
    pred_probs = model.predict(mel_input)
    pred_index = np.argmax(pred_probs)
    pred_genre = label_encoder.inverse_transform([pred_index])[0]

    st.success(f"✅ Predicted Genre: **{pred_genre.capitalize()}**")

    # Step 4: Fetch recommendations
    st.write("🎵 Fetching more similar songs for you...")

    jamendo_url = f"https://api.jamendo.com/v3.0/tracks/?client_id={JAMENDO_CLIENT_ID}&format=json&limit=10&tags={pred_genre.lower()}&include=musicinfo&audioformat=mp32"

    response = requests.get(jamendo_url)
    data = response.json()

    if 'results' in data and len(data['results']) > 0:
        for track in data['results']:
            st.subheader(track['name'])
            st.image(track['image'], width=150)
            st.write(f"Artist: {track['artist_name']}")
            st.audio(track['audio'])
    else:
        st.warning("❗ No recommendations found for this genre on Jamendo.")

