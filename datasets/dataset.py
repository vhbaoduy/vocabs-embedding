import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, Sampler
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


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

