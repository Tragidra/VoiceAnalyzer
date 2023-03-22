import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split

# загрузка и обработка (вариант с транспорнированием, без этого почему-то бьётся матрица)
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = librosa.util.fix_length(audio, size=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

df = pd.read_csv('test.csv')
X = []
y = []
for index, row in df.iterrows():
    file_path = row['file_path']
    label = row['Aggressive']
    mfccs = load_audio(file_path)
    X.append(mfccs)
    y.append(label)
X = np.array(X)
y = np.array(y)

# раскидываем информацию по сетам
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# простейшая модель
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(40,)), #первый слой, принимает 40 показателей частоты MFCC, 128 нейронов
    tf.keras.layers.Dropout(0.5), #половину входных данных обнуляем - спойлер: от переобучения это не спасло
    tf.keras.layers.Dense(1, activation='sigmoid') #определяем, агрессивный голос или нет
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#Пока вроде как Адама хватает для оптимального варианта, градиентный спуск подходит.
#Бинарная кросс-энтропия всё ещё единственный выход для бинарной классификации, мне это не нравится


# всё по классике
checkpoint_filepath = 'model_checkpoint.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data=(X_test, y_test), callbacks=[model_checkpoint_callback]) #обучение

# размер батча выше не уменьшать, не улучшило
model.load_weights(checkpoint_filepath)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Потери при обучении: {loss}')
print(f'Оценочная точность: {accuracy}')

model.save('truemodel.h5') # я решил не сохранять отдельно метрики, оставив их в модели.
# Все веса и смещения записаны в модели, зачем их отдельно хранить?
