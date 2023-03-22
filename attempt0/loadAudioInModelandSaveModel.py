import numpy as np
import csv
import os
import tensorflow as tf

audio_dir = "audio_samples"

output_dir = "predictions"

# возможно неправильная частота аудиозаписей? ещё подумать над тем, как сделать без обрезки файлов
sr = 22050
duration = 6

# метрики с либреса
features = ["mfcc", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "zero_crossing_rate"]

labels = {}
with open(os.path.join(output_dir, "primer.csv"), "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        labels[row[0]] = int(float(row[1]))

X = []
y = []
with open(os.path.join(output_dir, "primer.csv"), "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        filename = row[0]
        if filename in labels:
            label = labels[filename]
            feature_vector = [float(val) for val in row[1:]]
            X.append(feature_vector)
            y.append(label)

# преобразовываем матрицы признаков м вектор меток в np матрицы
X = np.array(X)
y = np.array(y)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X, y, epochs=100, batch_size=32)

model.save("models")