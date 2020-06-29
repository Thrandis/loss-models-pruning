import torch.nn as nn


class VGG(nn.Module):

    def __init__(self, dims, bn=False):
        super(VGG, self).__init__()
        self.features = self._build_net(dims, bn)
        if bn:
            m = [nn.Linear(512, 512, bias=False),
                 nn.BatchNorm1d(512),
                 nn.ReLU(inplace=True),
                 nn.Linear(512, 512, bias=False),
                 nn.BatchNorm1d(512),
                 nn.ReLU(inplace=True),
                 nn.Linear(512, 10, bias=True),
                 nn.BatchNorm1d(10)]
        else:
            m = [nn.Linear(512, 512, bias=True),
                 nn.ReLU(inplace=True),
                 nn.Linear(512, 512, bias=True),
                 nn.ReLU(inplace=True),
                 nn.Linear(512, 10, bias=True)]
        self.classifier = nn.Sequential(*m)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _build_net(self, dims, bn):
        modules = []
        in_dim = 3
        for i, dim in enumerate(dims):
            if dim == 'M':
                modules += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if bn:
                    modules += [nn.Conv2d(in_dim, dim, kernel_size=3, padding=1,
                                          bias=False),
                                nn.BatchNorm2d(dim),
                                nn.ReLU(inplace=True)]
                else:
                    modules += [nn.Conv2d(in_dim, dim, kernel_size=3, padding=1,
                                          bias=True),
                                nn.ReLU(inplace=True)]
                in_dim = dim
        return nn.Sequential(*modules)


def VGG11():
    dims = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGG(dims, False)


def VGG19():
    dims = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
            512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    return VGG(dims, False)


def VGG11BN():
    dims = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGG(dims, True)


def VGG19BN():
    dims = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
            512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    return VGG(dims, True)
