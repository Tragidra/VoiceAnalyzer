import librosa
import numpy as np
import csv
import os
import tensorflow as tf

audio_dir = "all_records"

output_dir = ""

sr = 22050
duration = 6

features = ["mfcc", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "zero_crossing_rate"]

model = tf.keras.models.load_model("models")


csv_file = open(os.path.join(output_dir, "predictions/predtech.csv"), "w", newline="")
writer = csv.writer(csv_file)


header_row = ["filename", "prediction"]
writer.writerow(header_row)


for file in os.listdir(audio_dir):
    if file.endswith(".wav"):

        y, _ = librosa.load(os.path.join(audio_dir, file), sr=sr, duration=duration)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

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

        feature_vector = np.concatenate([mfcc_mean, mfcc_std, [spectral_centroid_mean, spectral_centroid_std],
                                         [spectral_bandwidth_mean, spectral_bandwidth_std], [spectral_rolloff_mean,
                                                                                             spectral_rolloff_std],
                                         [zero_crossing_rate_mean, zero_crossing_rate_std]])
        feature_vector = (feature_vector - np.mean(feature_vector)) / np.std(feature_vector)

        prediction = model.predict(np.array([feature_vector]))
        label = int(prediction > 0.5)

        writer.writerow([file, label])

csv_file.close()