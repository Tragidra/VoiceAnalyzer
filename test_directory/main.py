import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load audio files and extract features
def extract_features(file_path):
    with open(file_path, "rb") as f:
        audio, sr = librosa.load(f)
    mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0)
    return features

# Load dataset and labels
angry_files = ["angry1.wav", "angry2.wav", "angry3.wav"]
calm_files = ["calm1.wav", "calm2.wav", "calm3.wav"]
X = [extract_features(f) for f in angry_files + calm_files]
y = [1] * len(angry_files) + [0] * len(calm_files)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
clf = SVC(kernel="linear")
clf.fit(X_train, y_train)

# Test classifier on new audio file
new_file = "test.wav"
new_features = extract_features(new_file)
prediction = clf.predict([new_features])[0]
if prediction == 1:
    print("The voice in the audio recording is angry.")
else:
    print("The voice in the audio recording is calm.")
