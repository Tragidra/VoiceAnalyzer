import subprocess

OPENSMLIE_PATH = "SMILExtract.exe"

CONFIG_FILE_PATH = "IS09_emotion.conf"

INPUT_FILE_PATH = "output10.wav"

OUTPUT_FILE_PATH = "res.csv"

command = f"{OPENSMLIE_PATH} -C {CONFIG_FILE_PATH} -I {INPUT_FILE_PATH} -csvoutput {OUTPUT_FILE_PATH}"

subprocess.run(command, shell=True)
