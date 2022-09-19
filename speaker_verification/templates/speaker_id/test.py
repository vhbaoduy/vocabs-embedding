import torchaudio
from speechbrain.pretrained import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="D:/speechbrain/templates/speaker_id/model")
# signal, fs =torchaudio.load('D:/speechbrain/templates/speaker_id/data/tensorflow-speech-recognition/train/0a9f9af7/0a9f9af7_bird_down.wav')

# # Compute speaker embeddings
# embeddings = classifier.encode_batch(signal)

# # Perform classification
# output_probs, score, index, text_lab = classifier.classify_batch(signal)

# # Posterior log probabilities
# print(output_probs)

# # Score (i.e, max log posteriors)
# print(score)

# # Index of the predicted speaker
# print(index)

# # Text label of the predicted speaker
# print(text_lab)

# import torchaudio
# from random import choice
# from glob import glob
# from speechbrain.pretrained import EncoderClassifier
# import numpy as np
# classTes = EncoderClassifier.from_hparams(source="/home/ubuntu/speechbrain/recipes/CommonLanguage/lang_id/tests/31031987")

# listaCA = glob("/home/ubuntu/audiosSpeechbrain/audios/ca/*/*.wav")
# listaEN = glob("/home/ubuntu/audiosSpeechbrain/audios/en/*/*.wav")
# listaES = glob("/home/ubuntu/audiosSpeechbrain/audios/es/*/*.wav")
# resTes = {"ca":{"ca":0,"en":0,"es":0},"en":{"ca":0,"en":0,"es":0},"es":{"ca":0,"en":0,"es":0}}

# def idioma(dic):
#     keys = list(dic.keys())
#     vals = list(dic.values())
#     res = vals.index(max(vals))
#     return keys[res],round(float(max(vals)),3)

# def resultados(dic,vals):
#     for k,v in vals.items():
#         dic[k][v] += 1
#     return dic
# for i in range(100):
#     fCA = choice(listaCA)
#     [vTesCA],_,_,[lTesCA] = classTes.classify_file(fCA)
#     fEN = choice(listaEN)
#     [vTesEN],_,_,[lTesEN] = classTes.classify_file(fEN)
#     fES = choice(listaES)
#     [vTesES],_,_,[lTesES] = classTes.classify_file(fES)
#     remove("./*.wav")
#     resTes = resultados(resTes,{"ca":lTesCA,"en":lTesEN,"es":lTesES})
#     print(resTes)
