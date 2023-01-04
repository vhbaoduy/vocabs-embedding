# 4 4 4
rm -rf data/out
python gendata_mnist_v2.py -n 4 4 4 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn_x3.yaml --device=cuda:0
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine_x3.py hparams/verification_ecapa.yaml > result_txt/4_x3_1.txt
mv results/ecapa_augment results/4_x3_1

rm -rf data/out
python gendata_mnist_v2.py -n 4 4 4 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn_x3.yaml --device=cuda:0
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine_x3.py hparams/verification_ecapa.yaml > result_txt/4_x3_2.txt
mv results/ecapa_augment results/4_x3_2

rm -rf data/out
python gendata_mnist_v2.py -n 4 4 4 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn_x3.yaml --device=cuda:0
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine_x3.py hparams/verification_ecapa.yaml > result_txt/4_x3_3.txt
mv results/ecapa_augment results/4_x3_3

# 0 0 0
rm -rf data/out
python gendata_mnist_v2.py -n 0 0 0 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn_x3.yaml --device=cuda:0
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine_x3.py hparams/verification_ecapa.yaml > result_txt/0_x3_1.txt
mv results/ecapa_augment results/0_x3_1

rm -rf data/out
python gendata_mnist_v2.py -n 0 0 0 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn_x3.yaml --device=cuda:0
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine_x3.py hparams/verification_ecapa.yaml > result_txt/0_x3_2.txt
mv results/ecapa_augment results/0_x3_2

rm -rf data/out
python gendata_mnist_v2.py -n 0 0 0 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn_x3.yaml --device=cuda:0
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine_x3.py hparams/verification_ecapa.yaml > result_txt/0_x3_3.txt
mv results/ecapa_augment results/0_x3_3

# 5 5 5
rm -rf data/out
python gendata_mnist_v2.py -n 5 5 5 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn_x3.yaml --device=cuda:0
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine_x3.py hparams/verification_ecapa.yaml > result_txt/5_x3_1.txt
mv results/ecapa_augment results/5_x3_1

rm -rf data/out
python gendata_mnist_v2.py -n 5 5 5 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn_x3.yaml --device=cuda:0
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine_x3.py hparams/verification_ecapa.yaml > result_txt/5_x3_2.txt
mv results/ecapa_augment results/5_x3_2

rm -rf data/out
python gendata_mnist_v2.py -n 5 5 5 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn_x3.yaml --device=cuda:0
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine_x3.py hparams/verification_ecapa.yaml > result_txt/5_x3_3.txt
mv results/ecapa_augment results/5_x3_3

# 2 2 2
rm -rf data/out
python gendata_mnist_v2.py -n 2 2 2 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn_x3.yaml --device=cuda:0
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine_x3.py hparams/verification_ecapa.yaml > result_txt/2_x3_1.txt
mv results/ecapa_augment results/2_x3_1

rm -rf data/out
python gendata_mnist_v2.py -n 2 2 2 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn_x3.yaml --device=cuda:0
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine_x3.py hparams/verification_ecapa.yaml > result_txt/2_x3_2.txt
mv results/ecapa_augment results/2_x3_2

rm -rf data/out
python gendata_mnist_v2.py -n 2 2 2 -dest_path data/out/train -times 1
python train_speaker_embeddings.py hparams/train_ecapa_tdnn_x3.yaml --device=cuda:0
mv results/ecapa_augment/1986/save/CKPT* results/ecapa_augment/1986/save/checkpoint
python speaker_verification_cosine_x3.py hparams/verification_ecapa.yaml > result_txt/2_x3_3.txt
mv results/ecapa_augment results/2_x3_3