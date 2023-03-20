import numpy as np
import pandas as pd
import librosa
import os

# Create a directory to store the generated audio files
if not os.path.exists('audio_samples'):
    os.makedirs('audio_samples')

# Load the audio files
aggressive_speech, sr = librosa.load('all_records/1_Busy.wav')
non_aggressive_speech, sr = librosa.load('all_records/0_Aggressive .wav')

# Trim the longer audio file to the length of the shorter one
min_length = min(len(aggressive_speech), len(non_aggressive_speech))
aggressive_speech = aggressive_speech[:min_length]
non_aggressive_speech = non_aggressive_speech[:min_length]

# Create white noise of the same length as the audio files
white_noise = np.random.randn(min_length)

# Combine the white noise with the speech to create noisy versions of the recordings
aggressive_noisy = aggressive_speech + white_noise
non_aggressive_noisy = non_aggressive_speech + white_noise

# Extract features from the audio recordings using librosa
aggressive_features = np.array(librosa.feature.mfcc(y=aggressive_noisy, sr=sr).mean(axis=1))
non_aggressive_features = np.array(librosa.feature.mfcc(y=non_aggressive_noisy, sr=sr).mean(axis=1))

# Combine the features into a DataFrame with labels
data = pd.DataFrame({'features': [aggressive_features, non_aggressive_features],
                     'label': ['aggressive', 'non-aggressive']})

# Save the data to a CSV file
data.to_csv('training_data.csv', index=False)
