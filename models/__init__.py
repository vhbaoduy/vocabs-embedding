from .resnet import *
from .classifer import *
from .resnext import *


def get_model(name, n_dims, l2_normalized, n_maps=None):
    assert name in ['resnet15', 'resnext']
    if name == 'resnet15':
        if n_maps is None:
            raise Exception("n_maps is not None")
        return Res15(n_maps, n_dims, l2_normalized=l2_normalized)

    if name == 'resnext':
        return  CifarResNeXt(n_dims=n_dims, l2_normalized=l2_normalized)