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

new_audio = load_audio('C:/Users/astra/Desktop/2023022623500336700870082042203201.wav')
new_audio = np.expand_dims(new_audio, axis=0)  # измерение батча

predicted_label = model.predict(new_audio)

if predicted_label > 0.5:
    print(predicted_label)
    print('Аудиозапись агрессивна.')
else:
    print(predicted_label)
    print('Никакой агрессии не обнаружено.')

#Пути к файлам для проверки
'''
Первые два файла результат правильный
C:/Users/astra/PycharmProjects/VoiceAnalyzer/attempt3/0_2023022620571669700870026942203201.wav   +
C:/Users/astra/PycharmProjects/VoiceAnalyzer/attempt3/0_2023022620514395500870021942203201.wav   +
C:/Users/astra/PycharmProjects/VoiceAnalyzer/attempt3/0_2023022621010693400870030042203201.wav   +
0_2023022622163954300870063342203201.wav    -
0_2023022622193574200870063942203201.wav    +
0_2023022622195580300870064342203201.wav    -
0_2023022623080680900870075942203201.wav    +
0_2023022623103512300870076342203201.wav    +
0_2023022623291638600870079642203201.wav    -
0_2023022623584689800870082642203201.wav    -
1_2023022620534654300870023342203201.wav    +
1_2023022622120670300870062042203201.wav    -
1_2023022622190923900870063742203201.wav    -
1_2023022622205745300870064542203201.wav    -
1_2023022622220907100870065042203201.wav    -
1_2023022623322112600870080242203201.wav    -
1_2023022623390402200870081042203201.wav    -
'''