# from speechbrain.pretrained import EncoderClassifier
# import torchaudio

# classifier = EncoderClassifier.from_hparams(source="model/", hparams_file='hparams_inference.yaml', savedir="model/")

# # audio_file = r'D:\tensorflow-speech-recognition-challenge\out_balance_1\\4f2ab70c\\4f2ab70c_three_down_1.wav'
# audio_file = r'D:\tensorflow-speech-recognition-challenge\train\audio\stop\4f2ab70c_nohash_0.wav'

# signal, fs = torchaudio.load(audio_file) 
# output_probs, score, index, text_lab = classifier.classify_batch(signal)
# print('Predicted: ' + text_lab[0])

# embeddings = classifier.encode_batch(signal)
# # output_probs, score, index, text_lab = classifier.classify_batch(signal)

# print(output_probs)
# print(score)
# print(index)
# print(text_lab)


from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="model/", hparams_file='hparams_inference.yaml', savedir="model/")

file1 = r'D:\tensorflow-speech-recognition-challenge\out_balance_1\\4f2ab70c\\4f2ab70c_three_down_1.wav'
file2 = r'D:\tensorflow-speech-recognition-challenge\out_balance_1\\4f2ab70c\\4f2ab70c_three_down_3.wav'

score, prediction = verification.verify_files(file1, file2)

print(score)
print(prediction) # True = same speaker, False=Different speakers