import librosa
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('Emotion_Voice_Detection_Model.h5')

labels_to_int = {'angry': 0, 'calm': 1, 'fearful': 2, 'happy': 3, 'sad': 4}

int_to_labels = {0: 'angry', 1: 'calm', 2: 'fearful', 3: 'happy', 4: 'sad'}

#запомнить - можно подавать только вав в 16-битном виде
audio_file = 'records/output10.wav'
signal, sr = librosa.load(audio_file, sr=16000)
mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)

window_size = 216
hop_length = 108
windows = []
for i in range(0, mfccs.shape[1] - window_size + 1, hop_length):
    window = mfccs[:, i:i+window_size]
    windows.append(window)

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
