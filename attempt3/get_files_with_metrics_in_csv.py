import os
import csv

directory = 'C:/Users/astra/PycharmProjects/VoiceAnalyzer/attempt3'

with open("test.csv", mode="w") as file:

    writer = csv.writer(file)

    writer.writerow(["file_path", "Aggressive"]) #запись идёт только названий, а не пути к файлу, надо переделать

    for filename in os.listdir(directory):

        if filename.endswith(".wav"):

            if filename[0] == "1":
                aggressive = 1
            else:
                aggressive = 0

            writer.writerow([filename, aggressive])
