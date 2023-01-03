# 4
rm -rf data/out
python gendata_mnist_v2.py -n 4 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --device=cuda:1
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine.py hparams/verification_ecapa.yaml > result_txt/4_x1_1.txt
mv results/ecapa_augment results/4_x1_1

rm -rf data/out
python gendata_mnist_v2.py -n 4 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --device=cuda:1
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine.py hparams/verification_ecapa.yaml > result_txt/4_x1_2.txt
mv results/ecapa_augment results/4_x1_2

rm -rf data/out
python gendata_mnist_v2.py -n 4 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --device=cuda:1
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine.py hparams/verification_ecapa.yaml > result_txt/4_x1_3.txt
mv results/ecapa_augment results/4_x1_3

# 0
rm -rf data/out
python gendata_mnist_v2.py -n 0 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --device=cuda:1
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine.py hparams/verification_ecapa.yaml > result_txt/0_x1_1.txt
mv results/ecapa_augment results/0_x1_1

rm -rf data/out
python gendata_mnist_v2.py -n 0 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --device=cuda:1
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine.py hparams/verification_ecapa.yaml > result_txt/0_x1_2.txt
mv results/ecapa_augment results/0_x1_2

rm -rf data/out
python gendata_mnist_v2.py -n 0 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --device=cuda:1
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine.py hparams/verification_ecapa.yaml > result_txt/0_x1_3.txt
mv results/ecapa_augment results/0_x1_3

# 5
rm -rf data/out
python gendata_mnist_v2.py -n 5 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --device=cuda:1
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine.py hparams/verification_ecapa.yaml > result_txt/5_x1_1.txt
mv results/ecapa_augment results/5_x1_1

rm -rf data/out
python gendata_mnist_v2.py -n 5 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --device=cuda:1
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine.py hparams/verification_ecapa.yaml > result_txt/5_x1_2.txt
mv results/ecapa_augment results/5_x1_2

rm -rf data/out
python gendata_mnist_v2.py -n 5 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --device=cuda:1
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine.py hparams/verification_ecapa.yaml > result_txt/5_x1_3.txt
mv results/ecapa_augment results/5_x1_3

# 2
rm -rf data/out
python gendata_mnist_v2.py -n 2 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --device=cuda:1
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine.py hparams/verification_ecapa.yaml > result_txt/2_x1_1.txt
mv results/ecapa_augment results/2_x1_1

rm -rf data/out
python gendata_mnist_v2.py -n 2 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --device=cuda:1
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine.py hparams/verification_ecapa.yaml > result_txt/2_x1_2.txt
mv results/ecapa_augment results/2_x1_2

rm -rf data/out
python gendata_mnist_v2.py -n 2 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn.yaml --device=cuda:1
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine.py hparams/verification_ecapa.yaml > result_txt/2_x1_3.txt
mv results/ecapa_augment results/2_x1_3