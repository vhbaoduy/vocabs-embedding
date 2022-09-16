import torch
import torch.nn as nn
import torch.functional as F
from .backbones import *


class Network(nn.Module):
    def __init__(self,
                 model_name,
                 embedding_dims,
                 l2_norm,
                 num_classes):
        super(Network, self).__init__()

        if model_name == 'resnet15':
            self.base = Res15(n_maps=45, n_dims=embedding_dims)
        elif model_name == 'resnet':
            self.base = CifarResNeXt(in_channels=1, n_dims=embedding_dims)

        self.num_classes = num_classes
        self.embedding_dims = embedding_dims
        self.model_name = model_name
        self.l2_norm = l2_norm
        self.classify = nn.Linear(embedding_dims, num_classes, bias=False)
        self.batch_norm = nn.BatchNorm1d(embedding_dims)

    def forward(self, x):
        feat = self.base(x)
        feat_inference = self.batch_norm(feat)
        scores = self.classify(feat_inference)
        if self.l2_norm:
            feat = F.normalize(feat, p=2, dim=1)
        return scores, feat
