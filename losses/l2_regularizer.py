import torch
import torch.nn.functional as F


def _regularize(embeddings):
    l2_embeds = F.normalize(embeddings, p=2, dim=-1)
    # l2_embeds.requires_grad = True
    return l2_embeds


class L2Regularizer(object):
    def __init__(self):
        pass

    def __call__(self, embeddings):
        l2_signal = _regularize(embeddings)
        return l2_signal
