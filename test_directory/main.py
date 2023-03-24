import os
import subprocess
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

opensmile_path = 'SMILExtract.exe'

config_path = 'IS09_emotion.conf'

train_data_path = 'train_data.csv'

test_data_path = 'test_data.csv'

column_names = ['file_name', 'class_label']

feature_names = ['mfcc[1-12]', 'mfcc[13-26]', 'mfcc[27-39]', 'spectral_contrast[1-7]', 'spectral_flux', 'loudness_ehs', 'voiced_segments']

num_trees = 100

aggression_threshold = 0.5

train_data = pd.read_csv(train_data_path, header=None, names=column_names)
train_features = []
train_labels = []
for i in range(len(train_data)):
    file_path = train_data.loc[i, 'file_name']
    class_label = train_data.loc[i, 'class_label']
    #сделать завтра скип хедера, из-за этого фейлится
    command = f'{opensmile_path} -C {config_path} -I {file_path} -csvoutput tmp.csv'
    subprocess.call(command, shell=True)
    features = pd.read_csv('tmp.csv', header=None, usecols=[*range(2, 72)], names=feature_names)
    train_features.append(features.values.reshape(-1))
    train_labels.append(class_label)
os.remove('tmp.csv')

rfc = RandomForestClassifier(n_estimators=num_trees)
rfc.fit(train_features, train_labels)

test_data = pd.read_csv(test_data_path, header=None, names=column_names)
test_features = []
test_labels = []
for i in range(len(test_data)):
    file_path = test_data.loc[i, 'file_name']
    class_label = test_data.loc[i, 'class_label']
    command = f'{opensmile_path} -C {config_path} -I {file_path} -csvoutput tmp.csv'
    subprocess.call(command, shell=True)
    features = pd.read_csv('tmp.csv', header=None, usecols=[*range(2, 72)], names=feature_names)
    test_features.append(features.values.reshape(-1))
    test_labels.append(class_label)
os.remove('tmp.csv')

predictions = rfc.predict_proba(test_features)[:, 1]
predicted_labels = np.where(predictions > aggression_threshold, 1, 0)
true_labels = np.array(test_labels)
accuracy = accuracy_score(true_labels, predicted_labels)

print('Результат:', accuracy)
