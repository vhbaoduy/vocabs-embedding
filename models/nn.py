import torch
import torch.nn as nn
import torch.functional as F
from .backbones import *


class Network(nn.Module):
    def __init__(self,
                 model_name,
                 embedding_dims,
                 l2_norm,
                 num_classes=None,
                 classify=False):
        super(Network, self).__init__()

        if model_name == 'resnet15':
            self.base = Res15(n_maps=45, n_dims=embedding_dims)
        elif model_name == 'resnet':
            self.base = CifarResNeXt(in_channels=1, n_dims=embedding_dims)
        elif model_name == 'bc_resnet':
            self.base = BcResNetModel(n_dims=embedding_dims)

        self.num_classes = num_classes
        self.embedding_dims = embedding_dims
        self.model_name = model_name
        self.l2_norm = l2_norm
        self.classify = classify
        if classify:
            self.classify_layer = nn.Linear(embedding_dims, num_classes, bias=False)

    def forward(self, x):
        feat = self.base(x)
        scores = None
        if self.classify:
            scores = self.classify_layer(feat)
        if self.l2_norm:
            feat = F.normalize(feat, p=2, dim=1)
        return scores, feat
