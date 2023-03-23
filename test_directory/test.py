import librosa
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('Emotion_Voice_Detection_Model.h5')

# Define the mapping of emotion labels to integers
labels_to_int = {'angry': 0, 'calm': 1, 'fearful': 2, 'happy': 3, 'sad': 4}

# Define the mapping of integers to emotion labels
int_to_labels = {0: 'angry', 1: 'calm', 2: 'fearful', 3: 'happy', 4: 'sad'}

# Load the audio file and extract its features
audio_file = 'records/2023022620493620200870020342203201.wav'
signal, sr = librosa.load(audio_file, sr=16000)
mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)

# Split the audio feature matrix into sliding windows
window_size = 216
hop_length = 108
windows = []
for i in range(0, mfccs.shape[1] - window_size + 1, hop_length):
    window = mfccs[:, i:i+window_size]
    windows.append(window)
#Здесь ведутся работы над использованием https://github.com/MiteshPuthran/Speech-Emotion-Analyzer для большей точности маркировки
windows = np.array(windows)
windows = np.expand_dims(windows, axis=-1)

predictions = []
for window in windows:
    print(np.expand_dims(window, axis=0))
    prediction = model.predict(np.expand_dims(window, axis=0))[0]
    predictions.append(prediction)

average_prediction = np.mean(predictions, axis=0)
predicted_label = int_to_labels[np.argmax(average_prediction)]

print('Анализ выдал результаты:', predicted_label)
