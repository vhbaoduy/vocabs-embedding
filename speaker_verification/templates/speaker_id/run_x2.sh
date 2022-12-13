rm -rf data/out
python gendata_mnist.py -first_word $1 -second_word $2
python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --device=cuda:2
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine.py hparams/verification_ecapa.yaml
