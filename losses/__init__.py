from .triplet import *
from .triplet_selector import *


def make_loss_fn(selective_type,
                 type_loss,
                 margin,
                 alpha,
                 beta):
    """
    :param beta:
    :param alpha:
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

    if type_loss == 'softmax_triplet':
        def loss_fn(scores, feat, target):
            loss, len_triplet = triplet_loss(feat, target)
            return alpha * loss + beta * F.cross_entropy(scores, target), len_triplet
    elif type_loss == 'triplet':
        def loss_fn(scores, feat, target):
            return triplet_loss(feat, target)
    return loss_fn
