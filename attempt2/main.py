import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split

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
    tf.keras.layers.Dense(128, activation='relu', input_shape=(40,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# всё по классике
checkpoint_filepath = 'model_checkpoint.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

model.fit(X_train, y_train, epochs=70, batch_size=32, validation_data=(X_test, y_test), callbacks=[model_checkpoint_callback])

model.load_weights(checkpoint_filepath)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Потери при обучении: {loss}')
print(f'Оценочная точность: {accuracy}')

model.save('truemodel.h5')
