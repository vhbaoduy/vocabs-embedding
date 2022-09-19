import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.manifold import TSNE
import utils
import os
import pandas as pd


def compute_distances(X):
    dists = -2. * np.matmul(X, np.transpose(X)) + np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(X ** 2,
                                                                                                 axis=1).reshape(1, -1)
    return dists


def compute_variance(features):
    mean = np.mean(features, axis=0)
    norm = features - mean
    variance = np.mean(norm ** 2)
    radius = np.max(np.linalg.norm(norm, axis=0))
    return variance, radius


def load_features(path: str):
    if os.path.isdir(path):
        vocab_name = path.split('\\')[-1]
        features = []
        for file_name in os.listdir(path):

            if file_name.endswith('.npy') and file_name != vocab_name + '_mean.npy':
                truth = file_name.split('_')[0]
                if truth == vocab_name:
                    feat = np.load(os.path.join(path, file_name))
                    features.append(feat)
        return vocab_name, np.array(features), len(features)


def find_max_min(path, labels):
    embed_mean = []
    for i in range(len(labels)):
        embed_mean.append(np.load(os.path.join(path, labels[i], labels[i] + '_mean.npy')))

    embed_mean = np.array(embed_mean)
    dists = np.abs(compute_distances(embed_mean))
    dists = dists[dists != 0]
    # print(mask)
    # print(dists.shape)
    # print(dists)
    max_value = np.max(dists)
    min_value = np.min(dists)
    max_idx = np.where(dists == max_value)
    min_idx = np.where(dists == min_value)
    print(min_idx)
    print(max_idx)
    # print(min_idx)
    print("Max: ", (labels[max_idx[0]], labels[max_idx[1]]))
    print("Min: ", (labels[min_idx[0]], labels[min_idx[1]]))


if __name__ == '__main__':
    configs = utils.load_config_file('./configs/configs.yaml')
    dataset_cfgs = configs['Dataset']
    labels = dataset_cfgs['labels']

    path_test = './output/test_infer/inferences'
    path_valid = './output/valid_infer/inferences_valid'

    # vocabs = []
    # variances = []
    # radius = []
    #
    # for label in labels:
    #     vocab, features, total = load_features(os.path.join(path_test, label))
    #     var, r = compute_variance(features)
    #     vocabs.append(vocab)
    #     variances.append(var)
    #     radius.append(r)
    # data = {
    #     'vocab': vocabs,
    #     'variance': variances,
    #     'radius': radius
    # }
    # df = pd.DataFrame(data)
    # df.to_csv('./output/result_test.csv', index=False)
    #
    # df['distance'] = 0.95 * df['variance'] + .05 * df['radius']
    # df_radius = df.sort_values(by='radius', ascending=False)
    # df_var = df.sort_values(by='variance', ascending=False)
    # df_temp = df.sort_values(by='distance', ascending=False).reset_index()
    #
    # print(df_radius.head(5))
    # print(df_var.head(5))
    # print(df_temp.head(5))

    vocabs = []
    variances = []
    radius = []

    for label in labels:
        vocab, features_1, total = load_features(os.path.join(path_valid, label))
        vocab, features_2, total = load_features(os.path.join(path_test, label))
        features = np.append(features_1, features_2, 0)
        var, r = compute_variance(features)
        vocabs.append(vocab)
        variances.append(var)
        radius.append(r)
    data = {
        'vocab': vocabs,
        'variance': variances,
        'radius': radius
    }
    df = pd.DataFrame(data)
    df.to_csv('./output/result_valid_test-truth.csv', index=False)

    df['distance'] = 0.95 * df['variance'] + .05 * df['radius']
    df_radius = df.sort_values(by='radius', ascending=False)
    df_var = df.sort_values(by='variance', ascending=False)
    df_temp = df.sort_values(by='distance', ascending=False).reset_index()

    print(df_radius.head(5))
    print(df_var.head(5))
    print(df_temp.head(5))

    # embeddings = np.array(embeddings)
    # tne_proj_list = []
    # for i in range(n_label):
    #     tsne = TSNE(3, verbose=1)
    #     tsne_proj = tsne.fit_transform(X=embeddings[i])
    #     tne_proj_list.append(tsne_proj)
    #
    # # Plot those points as a scatter plot and label them based on the pred labels
    # cmap = cm.get_cmap('tab20')
    # fig, ax = plt.subplots(figsize=(8, 8))
    # for i in range(n_label):
    #     ax.scatter(tne_proj_list[i][:, 0], tne_proj_list[i][:, 1],tne_proj_list[i][:,2], c=np.array(cmap(i)).reshape(1, 4), label=labels[i],
    #                alpha=0.5)
    # ax.legend(fontsize='large', markerscale=2)
    # plt.show()
