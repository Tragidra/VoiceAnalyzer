import os
import csv

directory = 'C:/Users/astra/PycharmProjects/VoiceAnalyzer/audio_samples'

with open("train_data.csv", mode="w") as file:

    writer = csv.writer(file)

    writer.writerow(["file_path", "label"]) #взято из второй версии нейросети

    for filename in os.listdir(directory):

        if filename.endswith(".wav"):

            if filename[0] == "1":
                aggressive = 'aggressive'
            else:
                aggressive = 'non-aggressive'

            writer.writerow([filename, aggressive])
