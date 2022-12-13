# Format data from word/speaker to speaker/word
import os
import re
import shutil
from collections import Counter
import wave
import random
import json
import argparse

speaker_sample = 50
vocab_sample = speaker_sample ** 2

speaker_num = 60
verification_num = speaker_sample
src_path="data/mnist/dataset"
words_file = []

def combine_data(words):
    speakers = []
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for word in words:
        matches = [re.match(r'^.*?_', i) for i in os.listdir(os.path.join(src_path, word))]
        words_file.append(Counter([i.group() for i in matches if i]))
    count = 0
    for i in words_file[0]:
        if count >= speaker_num:
            break
        speaker = i.replace("_","")
        # print(speaker)
        if words_file[0][i] < speaker_sample or words_file[1][i] < speaker_sample:
            continue
        if not os.path.exists(os.path.join(dest_path,speaker)):
            os.makedirs(os.path.join(dest_path,speaker))
        speakers.append(speaker)
        for c1 in range(speaker_sample):
 #           for c2 in range(speaker_sample):
                data = []
                outfile = os.path.join(dest_path, speaker+"/"+speaker+"_"+words[0]+"_"+words[1]+"_"+str(c1)+".wav")
                for word in words:
                    w = wave.open(os.path.join(src_path, word, speaker+"_NO_"+str(c1)+".wav"), 'rb')
                    data.append( [w.getparams(), w.readframes(w.getnframes())])
                    w.close()

        #        w = wave.open(os.path.join(src_path, words[1], speaker+"_NO_"+str(c1)+".wav"), 'rb')
         #       data.append( [w.getparams(), w.readframes(w.getnframes())] )
          #      w.close()
                    
                output = wave.open(outfile, 'wb')
                output.setparams(data[0][0])
                for i in range(len(data)):
                    output.writeframes(data[i][1])
                output.close()
        count += 1
    return speakers

def create_verification_list(speakers, times):
    path_file = "verification_" + times + ".txt"
    f = open(path_file, 'w')
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
                if str(j) != hash and speaker != ran_speaker[0]:
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

            w = wave.open(os.path.join(src_path, word, speaker+"_NO_"+str(c1)+".wav"), 'rb')
            data.append( [w.getparams(), w.readframes(w.getnframes())])
            w.close()
                
            output = wave.open(outfile, 'wb')
            output.setparams(data[0][0])
            for i in range(len(data)):
                output.writeframes(data[i][1])
            output.close()
        count += 1
    return speakers

def create_verification_list_single_word(speakers, times):
    path_file = "verification_" + times + ".txt"
    f = open(path_file, 'w')
    random.shuffle(speakers)
    for i in range(len(speakers)//2):
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
            
            ran_speaker_num = 0
            while ran_speaker_num < verification_num - 1:
                ran_speaker = random.sample(speakers, 1)
                j = random.randint(0, verification_num-1)
                file1=word
                file2=ran_speaker[0]+"_"+vocab+"_"+str(j)+".wav"
                if str(j) != hash and speaker != ran_speaker[0]:
                    f.write(str(0)+" "+speaker+"/"+file1+" "+ran_speaker[0]+"/"+file2+"\n")
                    ran_speaker_num += 1
    f.close()


# speakers = combine_data()
# create_verification_list(speakers)
parser = argparse.ArgumentParser('Generate data mnist')
parser.add_argument('-single_word', type=str, default=None)
parser.add_argument('-first_word', type=str, default=None)
parser.add_argument('-second_word', type=str, default=None)
parser.add_argument('-third_word', type=str, default=None)
parser.add_argument('-dest_path', type=str, default="data/out/train/")
parser.add_argument('-times', type=str, default=None)
args = parser.parse_args()
dest_path = args.dest_path
if args.single_word != None:
    speakers = create_data_single_word(args.single_word)
    create_verification_list_single_word(speakers, args.times)
else:
    words = []
    words.append(args.first_word)
    words.append(args.second_word)
    if args.third_word != None:
        words.append(args.third_word)
    speakers = combine_data(words)
    create_verification_list(speakers, args.times)