import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False

N_FFT = 512
N_CHANNELS = round(1 + N_FFT / 2)
OUT_CHANNELS = 32


class RandomCNN(nn.Module):
    def __init__(self):
        super(RandomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)

        # self.batch_norm6 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.batch_norm3 = nn.BatchNorm2d(256)
        # self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.batch_norm4 = nn.BatchNorm2d(256)
        # self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        # self.batch_norm5 = nn.BatchNorm2d(512)
        # self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        # self.batch_norm6 = nn.BatchNorm2d(512)
        # self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        # self.batch_norm7 = nn.BatchNorm2d(512)
        # self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        # self.batch_norm8 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        # x = self.relu(x)
        # x = self.max_pooling(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 4, 4)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)

        x = self.conv3(x)

        return x


# a_random = Variable(torch.randn(1, 1, 257, 430)).float()
# model = RandomCNN()
# a_O = model(a_random)
# print(a_O.shape)