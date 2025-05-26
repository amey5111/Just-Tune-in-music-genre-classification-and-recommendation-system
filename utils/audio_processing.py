# utils/audio_processing.py

import librosa
import numpy as np

def preprocess_audio(file_path, max_pad_len=1300):
    # Load audio
    y, sr = librosa.load(file_path, sr=22050)  # ensure same sample rate
    
    # Generate Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pad or trim
    if mel_db.shape[1] < max_pad_len:
        pad_width = max_pad_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :max_pad_len]

    # Normalize
    mel_db = (mel_db - np.mean(mel_db)) / np.std(mel_db)

    # Add channel dimension
    mel_db = mel_db[..., np.newaxis]

    return mel_db
