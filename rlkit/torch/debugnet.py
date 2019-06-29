import torch
import torch.nn as nn
import torch.nn.functional as F



class Debugnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=3)
        # self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 2)  #1st in_channel: color channels
        # self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 32, kernel_size = 3, stride = 2) #kernal_size: actually 3*3
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)  # kernal_size: actually 3*3

    def forward(self, t):
        t = self.fc1(t)
        t = F.relu(t)
        t = self.out(t)

        # t = self.conv1(t)
        # t = F.relu(t)
        # t = F.max_pool2d(t, kernel_size=2, stride=2)
        #
        # t = self.conv2(t)
        # t = F.relu(t)
        # t = F.max_pool2d(t, kernel_size=2, stride=2)
        #
        # t = self.conv3(t)
        # t = F.relu(t)
        # t = F.max_pool2d(t, kernel_size=2, stride=2)

        return t


