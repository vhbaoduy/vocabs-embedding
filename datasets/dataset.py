import os
import numpy as np
import torch
from torch.utils.data import Dataset

import utils


class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir, df, sample_rate, labels, transform=None):
        self.root_dir = root_dir
        self.classes = labels
        self.df = df
        self.sample_rate = sample_rate
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.root_dir, row['file_name'])
        label = utils.label_to_index(self.classes, row['vocab'])
        samples, sample_rate = utils.load_audio(path, self.sample_rate)
        data = {
            'samples': samples,
            'sample_rate': sample_rate,
            'target': label,
            'path': row['file_name']
        }
        if self.transform is not None:
            data = self.transform(data)
        return data

    def make_weights_for_balanced_classes(self):
        df_count = self.df.groupby(self.df['vocab'])['vocab'].count()
        N = sum(df_count)
        weight = N / np.array(df_count.loc[self.df['vocab']])
        return weight

    def get_labels(self):
        labels = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            labels.append(utils.label_to_index(self.classes, row['vocab']))

        return torch.LongTensor(labels)


class BackgroundNoiseDataset(Dataset):
    def __init__(self, path, transform, sample_rate, sample_length=1):
        noise_files = [file for file in os.listdir(path) if file.endswith('.wav')]
        samples = []
        for f in noise_files:
            noise_path = os.path.join(path, f)
            sample, sample_rate = utils.load_audio(noise_path, sample_rate)
            samples.append(sample)

        samples = np.hstack(samples)
        c = int(sample_rate * sample_length)
        r = len(samples) // c
        self.samples = samples[:r * c].reshape(-1, c)
        self.sample_rate = sample_rate
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        data = {
            'samples': self.samples[index],
            'sample_rate': self.sample_rate,
            'target': 1,
            'path': self.path
        }

        if self.transform is not None:
            data = self.transform(data)
        return data

#
# if __name__ == '__main__':
#     labels = [1,4,6,7,9]
#     t = torch.LongTensor(labels)
#     print(t)
