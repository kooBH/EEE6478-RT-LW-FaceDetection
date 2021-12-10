import torch

import torch.nn as nn

class FCR(nn.Module):
    def __init__(self):
        super(FCR, self).__init__()
        self.fc_1 = nn.Linear(1024, 512)
        self.bn_1 = nn.BatchNorm1d(512)
        self.fc_2 = nn.Linear(512, 256)
        self.bn_2 = nn.BatchNorm1d(256)
        self.fc_3 = nn.Linear(256, 2)
        self.acti_1 = nn.Tanh()
        self.acti_2 = nn.Sigmoid()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.acti_1(x)
        x = self.dropout(x)

        x = self.fc_2(x)
        x = self.bn_2(x)
        x = self.acti_1(x)
        x = self.dropout(x)

        x = self.fc_3(x)
        x = self.acti_2(x)
        return x