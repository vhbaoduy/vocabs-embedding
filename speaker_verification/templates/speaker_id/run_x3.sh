python gendata_mnist.py -first_word $1 -second_word $2 -third_word $3 -dest_path $4 -times $5
python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --device=$6
mv results/ecapa_augment2/1986/save/CKPT* results/ecapa_augment2/1986/save/checkpoint