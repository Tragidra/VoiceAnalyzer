import librosa
import numpy as np
import csv
import os

# Set the directory containing the audio recordings
audio_dir = "audio_samples"

# Set the directory to save the extracted features
output_dir = "predictions"

# Set the sampling rate and duration for audio analysis
sr = 22050
duration = 6

# Set the audio features to extract
features = ["mfcc", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "zero_crossing_rate"]

# Open the CSV file for writing
csv_file = open(os.path.join(output_dir, "primer.csv"), "w", newline="")
writer = csv.writer(csv_file)

# Write the header row to the CSV file
header_row = features + ["label"]
writer.writerow(header_row)

# Loop through all audio files in the directory
for file in os.listdir(audio_dir):
    if file.endswith(".wav"):
        # Load the audio file and extract the desired features
        y, _ = librosa.load(os.path.join(audio_dir, file), sr=sr, duration=duration)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

        # Calculate the mean and standard deviation of each feature
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_centroid_std = np.std(spectral_centroid)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_bandwidth_std = np.std(spectral_bandwidth)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        spectral_rolloff_std = np.std(spectral_rolloff)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        zero_crossing_rate_std = np.std(zero_crossing_rate)

        # Get the label (assuming the file name contains the label)
        label = int(file.split("_")[0])

        # Write the features and label to the CSV file
        feature_row = np.concatenate([mfcc_mean, mfcc_std, [spectral_centroid_mean, spectral_centroid_std],
                                      [spectral_bandwidth_mean, spectral_bandwidth_std], [spectral_rolloff_mean,
                                                                                          spectral_rolloff_std],
                                      [zero_crossing_rate_mean, zero_crossing_rate_std]])
        writer.writerow(np.concatenate([feature_row, [label]]))

# Close the CSV file
csv_file.close()
