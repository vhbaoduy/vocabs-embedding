from librosa import load
from librosa.util import fix_length
import librosa
import numpy as np
import soundfile as sff
import os
import json

# src_path="../../../../AudioMNIST/data"
src_path="../../../../dataset/audio_mnist"
mix_dest_path="data/mnist/dataset"
male_dest_path="data/mnist/dataset_male"
female_dest_path="data/mnist/dataset_female"

male_list =[]
female_list=[]
f = open(src_path+'/audioMNIST_meta.txt')
data = json.load(f)
for i in data:
    if data[i]['gender']=='male':
        male_list.append(i)
    else:
        female_list.append(i)

def create_new_data(dest_path, list_speaker):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for folder in os.listdir(src_path):
        if os.path.isdir(os.path.join(src_path, folder)):
            for file in os.listdir(os.path.join(src_path, folder)):
                word = file.split("_")[0]
                speaker = file.split("_")[1]
                if speaker not in list_speaker:
                    continue
                no = file.split("_")[2].split(".")[0]
            # print(word, speaker, no)
                if not os.path.exists(os.path.join(dest_path, word)):
                    os.makedirs(os.path.join(dest_path, word))

                sf = 16000 # sampling frequency of wav file
                required_audio_size = 1 # audio of size 2 second needs to be padded to 5 seconds
                audio, sf = load(os.path.join(src_path, folder, file), sr=sf, mono=True) # mono=True converts stereo audio to mono
                padded_audio = fix_length(audio, size=required_audio_size*sf) # array size is required_audio_size*sampling frequency
            # print(sf)

            # print('Array length before padding', np.shape(audio))
            # print('Audio length before padding in seconds', (np.shape(audio)[0]/sf))
            # print('Array length after padding', np.shape(padded_audio))
            # print('Audio length after padding in seconds', (np.shape(padded_audio)[0]/sf))

                sff.write(os.path.join(dest_path, word, speaker + "_NO_" + no +".wav"), padded_audio, sf)
    
create_new_data(female_dest_path, female_list)
# print(len(male_list))