import torch
import torch.nn as nn


class Classifier(torch.nn.Module):
    def __init__(self, n_classes, emb_dims, use_softmax=False, hidden_cfgs=None):
        super(Classifier,self).__init__()
        self.fc = None
        self.n_classes = n_classes
        self.emb_dims = emb_dims
        self.use_softmax = use_softmax
        if hidden_cfgs is not None:
            self.fc = [nn.Linear(emb_dims, hidden_cfgs[0])]
            for i in range(1, len(hidden_cfgs)):
                self.fc.append(nn.Linear(hidden_cfgs[i - 1], hidden_cfgs[i]))
            for i, layer in enumerate(self.fc):
                self.add_module("fc{}".format(i + 1), layer)
            self.classifier = nn.Linear(hidden_cfgs[-1], n_classes)
        else:
            self.classifier = nn.Linear(in_features=emb_dims, out_features=n_classes)

    def forward(self, x):
        if self.fc is not None:
            for i in range(len(self.fc)):
                x = getattr(self, "fc{}".format(i + 1))(x)
        out = self.classifier(x)
        if self.use_softmax:
            out = torch.softmax(out, dim=1)
        return out


# if __name__ == '__main__':
#     encoder = Classifier(35, 45, use_softmax=True, hidden_cfgs=[64,32])
#     embs = torch.rand((128, 45))
#     out = encoder(embs)
#     print(out)