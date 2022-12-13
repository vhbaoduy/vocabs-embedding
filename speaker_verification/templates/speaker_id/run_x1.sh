rm -rf data/out
python gendata_mnist_v2.py -n $1 -dest_path $2 -times $3
python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --device=$4
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine.py hparams/verification_ecapa.yaml
