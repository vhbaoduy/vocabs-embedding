from speechbrain.pretrained import EncoderClassifier
import torchaudio

classifier = EncoderClassifier.from_hparams(source="model/", hparams_file='hparams_inference.yaml', savedir="model/")

audio_file = r'D:\tensorflow-speech-recognition-challenge\out_balance\\3d53244b\\3d53244b_three_down_18.wav'
signal, fs = torchaudio.load(audio_file) 
output_probs, score, index, text_lab = classifier.classify_batch(signal)
print('Predicted: ' + text_lab[0])

embeddings = classifier.encode_batch(signal)
output_probs, score, index, text_lab = classifier.classify_batch(signal)

print(output_probs)
print(score)
print(index)
print(text_lab)