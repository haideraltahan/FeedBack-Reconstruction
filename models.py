import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

Tensor = torch.cuda.FloatTensor if True else torch.FloatTensor


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), 100))))
    z = sampled_z * std + mu
    return z


class Encoder(nn.Module):
    def conv_block(self, in_channel, outchannel, num=2):
        if num == 2:
            return nn.Sequential(
                torch.nn.Conv2d(in_channel, outchannel, 3, padding=1),
                torch.nn.LeakyReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                torch.nn.Conv2d(in_channel, outchannel, 3, padding=1),
                torch.nn.LeakyReLU(inplace=True),
            )

    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        # conv1
        self.conv1 = self.conv_block(3, 64)
        self.conv1_max = torch.nn.MaxPool2d(2, stride=2)

        # conv2
        self.conv2 = self.conv_block(64, 128)
        self.conv2_max = torch.nn.MaxPool2d(2, stride=2)

        # conv3
        self.conv3 = self.conv_block(128, 256, 3)
        self.conv3_max = torch.nn.MaxPool2d(2, stride=2)

        # conv4
        self.conv4 = self.conv_block(256, 512, 3)
        self.conv4_max = torch.nn.MaxPool2d(2, stride=2)

        # conv5
        self.conv5 = self.conv_block(512, 512, 3)
        self.conv5_max = torch.nn.MaxPool2d(2, stride=2)

        self.last_encode = nn.Linear(512 * 7 * 7, 512)

        self.mu = nn.Linear(512, 100)
        self.logvar = nn.Linear(512, 100)

    def forward(self, img, **kwargs):
        activation = []
        x = self.conv1(img)
        activation.append(torch.flatten(x, 1))
        x = self.conv1_max(x)

        x = self.conv2(x)
        activation.append(torch.flatten(x, 1))
        x = self.conv2_max(x)

        x = self.conv3(x)
        activation.append(torch.flatten(x, 1))
        x = self.conv3_max(x)

        x = self.conv4(x)
        activation.append(torch.flatten(x, 1))
        x = self.conv4_max(x)

        x = self.conv5(x)
        activation.append(torch.flatten(x, 1))
        x = self.conv5_max(x)

        x = self.last_encode(torch.flatten(x, 1))
        activation.append(torch.flatten(x, 1))
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        activation.append(torch.flatten(z, 1))
        return z, activation


class Decoder(nn.Module):
    def deconv_block(self, in_channel, outchannel, num=2):
        if num == 2:
            return nn.Sequential(
                torch.nn.ConvTranspose2d(in_channel, outchannel, 2, stride=2, padding=0),
                torch.nn.LeakyReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                torch.nn.ConvTranspose2d(in_channel, outchannel, 2, stride=2, padding=0),
                torch.nn.LeakyReLU(inplace=True),
            )

    def __init__(self, **kwargs):
        super(Decoder, self).__init__()
        # DeConv1
        self.deconv1 = self.deconv_block(512, 512)
        # deconv2
        self.deconv2 = self.deconv_block(512, 256)
        # deconv3
        self.deconv3 = self.deconv_block(256, 128, 3)
        # deconv4
        self.deconv4 = self.deconv_block(128, 64, 3)
        # deconv5
        self.deconv5 = self.deconv_block(64, 3, 3)
        self.dencode = nn.Linear(100, 512 * 7 * 7)
        self.tan = nn.Tanh()

    def forward(self, z, **kwargs):
        activation = []
        x = self.dencode(z)
        activation.append(torch.flatten(x, 1))
        x = x.view(x.size()[0], 512, 7, 7)
        x = self.deconv1(x)
        activation.append(torch.flatten(x, 1))
        x = self.deconv2(x)
        activation.append(torch.flatten(x, 1))
        x = self.deconv3(x)
        activation.append(torch.flatten(x, 1))
        x = self.deconv4(x)
        activation.append(torch.flatten(x, 1))
        x = self.deconv5(x)
        activation.append(torch.flatten(x, 1))
        return self.tan(x), activation


class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(100 + 4, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity
