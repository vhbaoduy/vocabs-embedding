# Format data from word/speaker to speaker/word
import os
import re
import shutil
from collections import Counter
import wave
import random
import json
import csv
import pandas as pd

speaker_sample = 5
vocab_sample = speaker_sample ** 2

speaker_num = 25
verification_num = vocab_sample
src_path="data/speech_commands_v0.02"
dest_path="data/out/train/"
words = ['five', 'four']
single_word = "yes"
words_file = []

def read_info():
    f = open('info.json')
    data = json.load(f)
    return data

def read_csv():
    df = pd.read_csv('data_filter_5.csv')
    list_speaker = list(df["speaker"])
    list_file = list(df["file"])
    list_word = list(df["word"])
    res = []
    tmp=[]
    for index, i in enumerate(list_file):
        if list_word[index] not in words:
            continue
        tmp.append(i)
        if len(tmp) == speaker_sample:
            res.append((list_speaker[index], tmp))
            tmp=[]
    return res
        
def combine_data(list_file):
    speakers = []
    # data_info = read_info()
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    count = 0
    for i in range(speaker_num):
        speaker = list_file[i][0]
        list_speaker_1 = list_file[i][1]
        list_speaker_2 = list_file[i+vocab_sample][1]
        # print(list_speaker_1)
        # print(list_speaker_2)
        print(speaker)
        if not os.path.exists(os.path.join(dest_path,speaker)):
            os.makedirs(os.path.join(dest_path,speaker))
        speakers.append(speaker)
        for c1 in range(speaker_sample):
            for c2 in range(speaker_sample):
                data = []
                outfile = os.path.join(dest_path, speaker+"/"+speaker+"_"+words[0]+"_"+words[1]+"_"+str((c1*speaker_sample)+c2)+".wav")

                w = wave.open(os.path.join(src_path, list_speaker_1[c1]), 'rb')
                data.append( [w.getparams(), w.readframes(w.getnframes())])
                w.close()

                w = wave.open(os.path.join(src_path, list_speaker_2[c2]), 'rb')
                data.append( [w.getparams(), w.readframes(w.getnframes())] )
                w.close()
                    
                output = wave.open(outfile, 'wb')
                output.setparams(data[0][0])
                for i in range(len(data)):
                    output.writeframes(data[i][1])
                output.close()
        count += 1
    return speakers

def create_verfication_list_not_duplicate(speakers):
    f = open("verification.txt", 'w')
    random.shuffle(speakers)
    list_verify = []
    for i in range(len(speakers)//2):
        speaker = speakers[i]
        for count, word in enumerate(os.listdir(os.path.join(dest_path, speaker))):
            if count >= verification_num:
                break
            vocab = word.split("_")[1] + "_" + word.split("_")[2]
            hash = word.split("_")[3].split(".")[0]
            # print(vocab, speaker, hash)
            true_verify = 0
            for j in range(verification_num):
                file1=word
                file2=speaker+"_"+vocab+"_"+str(j)+".wav"
                if str(j) != hash and (file1,file2) not in list_verify and (file2,file1) not in list_verify:
                    f.write(str(1)+" "+speaker+"/"+file1+" "+speaker+"/"+file2+"\n")
                    list_verify.append((file1,file2))
                    true_verify += 1
            
            ran_speaker_num = 0
            while ran_speaker_num < true_verify:
                ran_speaker = random.sample(speakers, 1)
                j = random.randint(0, verification_num-1)
                file1=word
                file2=ran_speaker[0]+"_"+vocab+"_"+str(j)+".wav"
                if str(j) != hash and (file1,file2) not in list_verify and (file2,file1) not in list_verify:
                    f.write(str(0)+" "+speaker+"/"+file1+" "+ran_speaker[0]+"/"+file2+"\n")
                    list_verify.append((file1,file2))
                    ran_speaker_num += 1
    f.close()

def create_verfication_list(speakers):
    f = open("verification.txt", 'w')
    random.shuffle(speakers)
    for i in range(len(speakers)//2):
        speaker = speakers[i]
        for count, word in enumerate(os.listdir(os.path.join(dest_path, speaker))):
            if count >= verification_num:
                break
            vocab = word.split("_")[1] + "_" + word.split("_")[2]
            hash = word.split("_")[3].split(".")[0]
            # print(vocab, speaker, hash)

            for j in range(verification_num):
                file1=word
                file2=speaker+"_"+vocab+"_"+str(j)+".wav"
                if str(j) != hash:
                    f.write(str(1)+" "+speaker+"/"+file1+" "+speaker+"/"+file2+"\n")
            
            ran_speaker_num = 0
            while ran_speaker_num < verification_num - 1:
                ran_speaker = random.sample(speakers, 1)
                j = random.randint(0, verification_num-1)
                file1=word
                file2=ran_speaker[0]+"_"+vocab+"_"+str(j)+".wav"
                if str(j) != hash:
                    f.write(str(0)+" "+speaker+"/"+file1+" "+ran_speaker[0]+"/"+file2+"\n")
                    ran_speaker_num += 1
    f.close()


def create_data_single_word(word):
    speakers = []
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    matches = [re.match(r'^.*?_', i) for i in os.listdir(os.path.join(src_path, word))]
    words_file.append(Counter([i.group() for i in matches if i]))

    count = 0
    for i in words_file[0]:
        if count >= speaker_num:
            break
        speaker = i.replace("_","")
        # print(speaker)
        if words_file[0][i] < speaker_sample:
            continue
        if not os.path.exists(os.path.join(dest_path,speaker)):
            os.makedirs(os.path.join(dest_path,speaker))
        speakers.append(speaker)
        for c1 in range(speaker_sample):
            data = []
            outfile = os.path.join(dest_path, speaker+"/"+speaker+"_"+word+"_"+str(c1)+".wav")

            w = wave.open(os.path.join(src_path, word, speaker+"_nohash_"+str(c1)+".wav"), 'rb')
            data.append( [w.getparams(), w.readframes(w.getnframes())])
            w.close()
                
            output = wave.open(outfile, 'wb')
            output.setparams(data[0][0])
            for i in range(len(data)):
                output.writeframes(data[i][1])
            output.close()
        count += 1
    return speakers


def create_verfication_list_single_word(speakers):
    f = open("verification.txt", 'w')
    random.shuffle(speakers)
    for i in range(len(speakers)//3):
        speaker = speakers[i]
        for count, word in enumerate(os.listdir(os.path.join(dest_path, speaker))):
            if count >= verification_num:
                break
            vocab = word.split("_")[1]
            hash = word.split("_")[2].split(".")[0]

            for j in range(verification_num):
                file1=word
                file2=speaker+"_"+vocab+"_"+str(j)+".wav"
                if str(j) != hash:
                    f.write(str(1)+" "+speaker+"/"+file1+" "+speaker+"/"+file2+"\n")

            ran_speaker_num = 0
            while ran_speaker_num < verification_num - 1:
                ran_speaker = random.sample(speakers, 1)
                j = random.randint(0, verification_num-1)
                file1=word
                file2=ran_speaker[0]+"_"+vocab+"_"+str(j)+".wav"
                if str(j) != hash:
                    f.write(str(0)+" "+speaker+"/"+file1+" "+ran_speaker[0]+"/"+file2+"\n")
                    ran_speaker_num += 1
    f.close()


list_file = read_csv()
speakers = combine_data(list_file)
create_verfication_list(speakers)

# speakers = create_data_single_word(single_word)
# create_verfication_list_single_word(speakers)
