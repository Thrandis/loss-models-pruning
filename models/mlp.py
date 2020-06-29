import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_300_100(nn.Module):

    def __init__(self, input_size=28*28, num_classes=10, tanh=True):
        super(MLP_300_100, self).__init__()
        self.fc1 = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)
        if tanh:
            self.activation = torch.tanh
        else:
            self.activation = F.relu

    def forward(self, x): 
        out = x.view(x.size(0), -1) 
        out = self.activation(self.fc1(out))
        out = self.activation(self.fc2(out))
        out = self.fc3(out)
        return out 
