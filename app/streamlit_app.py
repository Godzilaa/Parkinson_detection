import streamlit as st
import joblib
import numpy as np
import librosa

# Set Streamlit page configuration
st.set_page_config(
    page_title="Parkinson's Detection",
    layout="centered",
    initial_sidebar_state="collapsed",
    page_icon="🧠"
)

# Apply dark theme
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    .stButton > button {
        background-color: #333;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model
model_path = "../model/nn_model.pkl"
model = joblib.load(model_path)

# Title
st.title("Parkinson's Detection")
st.write("Upload an audio file to check for Parkinson's disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

def extract_advanced_features(audio, sr):
    # Extract jitter and shimmer using librosa
    jitter = np.mean(np.abs(np.diff(audio)))
    shimmer = np.std(np.abs(np.diff(audio)))

    # Extract Harmonic-to-Noise Ratio (HNR) using librosa
    hnr = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))

    # Placeholder for TQWT and Formants (requires additional libraries or custom implementation)
    tqwt_features = [0] * 10  # Replace with actual TQWT feature extraction
    formant_features = [0] * 5  # Replace with actual formant extraction

    return [jitter, shimmer, hnr] + tqwt_features + formant_features

if uploaded_file is not None:
    # Load audio file
    try:
        audio, sr = librosa.load(uploaded_file, sr=None)
        st.audio(uploaded_file, format='audio/wav')

        # Extract multiple audio features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=12, fmin=20)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6, fmin=20)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        rms = librosa.feature.rms(y=audio)

        # Extract advanced features
        advanced_features = extract_advanced_features(audio, sr)

        # Aggregate features (mean and standard deviation)
        features = np.hstack([
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(chroma, axis=1), np.std(chroma, axis=1),
            np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1),
            np.mean(zero_crossing_rate), np.std(zero_crossing_rate),
            np.mean(rms), np.std(rms),
            advanced_features
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        # Display results
        if prediction[0] == 1:
            st.error(f"High likelihood of Parkinson's disease (Confidence: {probability:.2%})")
        else:
            st.success(f"Low likelihood of Parkinson's disease (Confidence: {1 - probability:.2%})")

    except Exception as e:
        st.error(f"Error processing audio file: {e}")