import torch
import numpy as np
from itertools import combinations
import torch.nn.functional as F
import torch.nn as nn


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


if __name__ == '__main__':
    pass
    # embs = torch.rand((128,45), requires_grad=False)
    # print(embs)
    # l2_re = L2Regularizer()
    # # labels = []
    # # for i in range(8):
    # #     labels.extend([i]*16)
    # # labels = np.array(labels)
    # # labels = torch.LongTensor(labels)
    # # triplets_fn = OnlineTripletLoss(0.5, 'hardest', False)
    # # loss = triplets_fn(embs, labels)
    # embs = l2_re(embs)
    # print(embs)
    # print(embs.size())
