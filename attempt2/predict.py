import tensorflow as tf
import librosa
import numpy as np

# Взято из оригинальной модели для обучения и порезано (ещё самая первая реализация в неправильной сети)
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = librosa.util.fix_length(audio, size=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

model = tf.keras.models.load_model('truemodel.h5')

new_audio = load_audio('C:/Users/astra/PycharmProjects/VoiceAnalyzer/audio_samples/0_DC_n30 (8).wav') #Аудиозапись с агрессией в голосе
new_audio = np.expand_dims(new_audio, axis=0)  # измерение батча

predicted_label = model.predict(new_audio)

if predicted_label > 0.5:
    print(predicted_label)
    print('Аудиозапись агрессивна.')
else:
    print(predicted_label)
    print('Никакой агрессии не обнаружено.')
