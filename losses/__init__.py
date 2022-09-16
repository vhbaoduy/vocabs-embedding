from .triplet import *
from .triplet_selector import *


def make_loss_fn(selective_type,
                 margin,
                 alpha):
    """
    :param selective_type:
    :param margin:
    :return:
    """
    assert selective_type in ["hardest", "random_hard", "semi_hard"]
    if selective_type == 'hardest':
        triplet_selector = HardestNegativeTripletSelector(margin=margin)
    elif selective_type == 'random_hard':
        triplet_selector = RandomNegativeTripletSelector(margin=margin)
    elif selective_type == 'semi_hard':
        triplet_selector = HardestNegativeTripletSelector(margin)

    triplet_loss = OnlineTripletLoss(margin=margin, triplet_selector=triplet_selector)

    def loss_fn(scores, feat, target):
        loss, len_triplet = triplet_loss(feat, target)
        return alpha * loss + (1-alpha) * F.cross_entropy(scores, target), len_triplet

    return loss_fn
