import torch
import numpy as np
from itertools import combinations
import torch.nn.functional as F


class OnlineTripletLoss(object):
    def __init__(self, margin=1.,
                 selection_type='random_hard',
                 use_gpu=True):
        self.margin = margin
        self.negative_selection = selection_type
        self.use_gpu = use_gpu

    def __call__(self, embeddings, labels):
        triplets = _get_triplets(embeddings,
                                 labels,
                                 type_fn=self.negative_selection,
                                 margin=self.margin)

        if self.use_gpu:
            triplets = triplets.to('cuda')
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)
        return losses.mean()


def _get_pairwise_distance_matrix(embeddings, squared=False):
    """
    Calculate distance matrix with L2
    :param squared:
    :param embeddings: Shape: batch_size x dims
    :return: Distance matrix: batch_size x batch_size
    """
    distance_matrix = -2 * embeddings.mm(torch.t(embeddings)) + \
                      embeddings.pow(2).sum(dim=1).view(1, -1) + \
                      embeddings.pow(2).sum(dim=1).view(-1, 1)

    # Distance always >= 0
    distance_matrix = F.relu(distance_matrix)
    if not squared:
        # Add epsilon to avoid the gradient of sqrt = 0
        mask = torch.eq(distance_matrix, 0).float()
        distance_matrix = distance_matrix + mask * 1e-16
        distance_matrix = torch.sqrt(distance_matrix)
        distance_matrix = distance_matrix * (1 - mask)

    return distance_matrix


# Choice negative
def _hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def _random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def _get_negative_selection_fn(type_fn):
    assert type_fn in ['hardest', 'random_hard']
    if type_fn == 'hardest':
        return _hardest_negative
    if type_fn == 'random_hard':
        return _random_hard_negative


def _get_triplets(embeddings, labels, type_fn='hardest', margin=1., use_cpu=True):
    negative_selection_fn = _get_negative_selection_fn(type_fn)
    if use_cpu:
        embeddings = embeddings.cpu()
        labels = labels.cpu().data.numpy()

    distance_matrix = _get_pairwise_distance_matrix(embeddings)
    distance_matrix = distance_matrix.cpu().data.numpy()
    triplets = []
    anchor_positive = None
    for label in set(labels):
        mask = (labels == label)
        label_indices = np.where(mask)[0]
        if len(label_indices) < 2:
            continue

        negative_indices = np.where(np.logical_not(mask))[0]
        anchor_positives = np.array(list(combinations(label_indices, 2)))  # All pairs
        ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]

        for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
            # print(ap_distance.shape)
            loss_values = ap_distance - distance_matrix[
                anchor_positive[0],
                negative_indices
            ] + margin
            # loss_values = loss_values.cpu().data.numpy()
            idx = negative_selection_fn(loss_values)
            if idx is not None:
                hard_negative = negative_indices[idx]
                triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

    # Check triplets
    if len(triplets) == 0:
        triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

    triplets = np.array(triplets)
    return torch.LongTensor(triplets)


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
