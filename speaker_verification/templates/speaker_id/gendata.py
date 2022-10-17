# Format data from word/speaker to speaker/word
import os
import re
import shutil
from collections import Counter
import wave
import random
import json

speaker_sample = 3
vocab_sample = speaker_sample ** 2

speaker_num = 67
verification_num = 5
src_path="data/speech_commands_v0.02/"
dest_path="data/out/train/"
words = ['six', 'six']
single_word = "yes"
words_file = []

validate_txt = src_path + "testing_list.txt"

def read_info():
    f = open('info.json')
    data = json.load(f)
    return data

def combine_data():
    speakers = []
    data_info = read_info()
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    print(len(data_info['speakers']))
    for word in words:
        matches = [re.match(r'^.*?_', i) for i in os.listdir(os.path.join(src_path, word))]
        words_file.append(Counter([i.group() for i in matches if i]))
    count = 0
    for i in words_file[0]:
        if count >= speaker_num:
            break
        speaker = i.replace("_","")
        # print(speaker)
        if words_file[0][i] < speaker_sample or words_file[1][i] < speaker_sample or speaker not in data_info['speakers']:
            continue
        if not os.path.exists(os.path.join(dest_path,speaker)):
            os.makedirs(os.path.join(dest_path,speaker))
        speakers.append(speaker)
        for c1 in range(speaker_sample):
            for c2 in range(speaker_sample):
                data = []
                outfile = os.path.join(dest_path, speaker+"/"+speaker+"_"+words[0]+"_"+words[1]+"_"+str((c1*speaker_sample)+c2)+".wav")

                w = wave.open(os.path.join(src_path, words[0], speaker+"_nohash_"+str(c1)+".wav"), 'rb')
                data.append( [w.getparams(), w.readframes(w.getnframes())])
                w.close()

                w = wave.open(os.path.join(src_path, words[1], speaker+"_nohash_"+str(c2)+".wav"), 'rb')
                data.append( [w.getparams(), w.readframes(w.getnframes())] )
                w.close()
                    
                output = wave.open(outfile, 'wb')
                output.setparams(data[0][0])
                for i in range(len(data)):
                    output.writeframes(data[i][1])
                output.close()
        count += 1
    return speakers

def create_verfication_list(speakers):
    f = open("verification.txt", 'w')
    random.shuffle(speakers)
    for i in range(len(speakers)//3):
        speaker = speakers[i]
        for count, word in enumerate(os.listdir(os.path.join(dest_path, speaker))):
            if count >= verification_num:
                break
            vocab = word.split("_")[1] + "_" + word.split("_")[2]
            hash = word.split("_")[3].split(".")[0]
            # print(vocab, speaker, hash)
            for i in range(verification_num):
                if str(i) != hash:
                    f.write(str(1)+" "+speaker+"/"+word+" "+speaker+"/"+speaker+"_"+vocab+"_"+str(i)+".wav\n")
            
            ran_speaker = []
            while len(ran_speaker) < verification_num:
                tmp = random.sample(speakers, 1)
                if tmp[0] not in ran_speaker and tmp[0] != speaker:
                    ran_speaker.append(tmp[0])
            for i in range(verification_num):
                if str(i) != hash:
                    f.write(str(0)+" "+speaker+"/"+word+" "+ran_speaker[i]+"/"+ran_speaker[i]+"_"+vocab+"_"+str(i)+".wav\n")
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
            # print(vocab, speaker, hash)
            for i in range(verification_num):
                if str(i) != hash:
                    f.write(str(1)+" "+speaker+"/"+word+" "+speaker+"/"+speaker+"_"+vocab+"_"+str(i)+".wav\n")
            
            ran_speaker = []
            while len(ran_speaker) < verification_num:
                tmp = random.sample(speakers, 1)
                if tmp[0] not in ran_speaker and tmp[0] != speaker:
                    ran_speaker.append(tmp[0])
            for i in range(verification_num):
                if str(i) != hash:
                    f.write(str(0)+" "+speaker+"/"+word+" "+ran_speaker[i]+"/"+ran_speaker[i]+"_"+vocab+"_"+str(i)+".wav\n")
    f.close()


speakers = combine_data()
create_verfication_list(speakers)

# speakers = create_data_single_word(single_word)
# create_verfication_list_single_word(speakers)
