import torch
from torch import nn
from torch.nn import functional as F

import utils


class Res15(nn.Module):
    def __init__(self, n_maps, n_dims):
        super().__init__()
        self.n_maps = n_maps
        self.n_dims = n_dims
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)
        self.n_layers = n_layers = 13
        dilation = True
        if dilation:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2 ** (i // 3)), dilation=int(2 ** (i // 3)),
                                    bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                                    bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)

        self.add_module("fc", nn.Linear(n_maps, n_dims))

    def forward(self, audio_signal):
        x = audio_signal.unsqueeze(1)
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        x = x.view(x.size(0), x.size(1), -1)  # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        x = getattr(self, "fc")(x)
        x = nn.PReLU()(x)
        # x = x.unsqueeze()
        # print(x.size())
        return x

    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad = False


if __name__ == '__main__':
    model = Res15(45, 128)
    audio = torch.rand((128, 32, 32), requires_grad=False)
    # feat = model(audio)
    # print(feat.size())
    # print(model)
    # samples, sample_rate = utils.load_audio("F:\\Datasets\\speech_commands_v0.02\\house\\00b01445_nohash_0.wav", 16000)
    # s = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_mels=64,n_fft=512)
    # print(s.shape)
    # model
