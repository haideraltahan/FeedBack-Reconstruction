import torch
import torch.nn as nn
from torch.nn import functional as F


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class conv_block(nn.Module):
    def __init__(self, in_channel, outchannel, kernel_size=3, padding=1, stride=2):
        super().__init__()
        self.feed = nn.Sequential(
            nn.Conv2d(in_channel, outchannel, kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(inplace=True),
            # nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        return self.feed(x)


class deconv_block(nn.Module):
    def __init__(self, in_channel, outchannel, kernel_size=3, stride=2, padding=0, act=True):
        super(deconv_block, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(in_channel, outchannel, kernel_size, stride=stride, padding=padding)
        self.activation = torch.nn.LeakyReLU(0.2, inplace=True)
        self.act = act

    def forward(self, x):
        if self.act:
            return self.activation(self.deconv(x))
        else:
            return self.deconv(x)


class Classifier(nn.Module):
    def __init__(self, ch=32, init=True, n_classes=4):
        super(Classifier, self).__init__()

        self.features = nn.Sequential(
            conv_block(3, 2 * ch, stride=1),
            conv_block(2 * ch, 2 * ch),
            conv_block(2 * ch, 4 * ch, stride=1),
            conv_block(4 * ch, 4 * ch),
            conv_block(4 * ch, 8 * ch, stride=1),
        )

        self.last_encode1 = nn.Linear(16 * ch, 8 * ch)
        self.last_encode2 = nn.Linear(8 * ch, n_classes)
        self.leak1 = nn.LeakyReLU(inplace=True)
        if init:
            init_model(self)

    def forward(self, img, get_mu_logvar=False, get_activations=False, get_before_act_func=True, layer=-1):
        x = self.features(img)
        x = self.leak1(self.last_encode1(torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)))
        return self.last_encode2(x)


class Encoder(nn.Module):
    def __init__(self, Tensor, intermediate_size=512, reparm=True, latent_dim=100, ch=32, init=True, return_mu=False):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.Tensor = Tensor
        self.reparm = reparm
        self.return_mu = return_mu

        self.features = nn.Sequential(
            conv_block(3, 2 * ch),
            conv_block(2 * ch, 4 * ch),
            conv_block(4 * ch, 8 * ch),
            conv_block(8 * ch, 16 * ch),
            conv_block(16 * ch, 16 * ch),
        )

        if reparm:
            self.last_encode = nn.Linear(512, intermediate_size)
            self.mu = nn.Linear(intermediate_size, latent_dim)
            self.logvar = nn.Linear(intermediate_size, latent_dim)
        else:
            self.last_encode = nn.Linear(512, latent_dim)
        print(reparm)
        self.leak = nn.LeakyReLU(inplace=True)

    def forward(self, img, get_mu_logvar=False, get_activations=False, get_before_act_func=True, layer=-1):
        x = self.features(img)
        x = self.last_encode(F.adaptive_avg_pool2d(x, (1, 1)).view(x.shape[0], -1))
        if self.reparm:
            x = self.leak(x)
            mu = self.mu(x)
            logvar = self.logvar(x)
            if self.return_mu:
                return reparameterize(mu, logvar), mu, logvar
            else:
                return reparameterize(mu, logvar)
        else:
            return x


class Decoder(nn.Module):

    def __init__(self, prior=False, m=2, latent_dim=100, n_classes=4, ch=32, deconv=False, init=True, **kwargs):
        super(Decoder, self).__init__()
        self.m = m
        self.ch = ch
        deconv = deconv_block
        self.encode = nn.Linear(latent_dim + (n_classes if prior else 0), 16 * ch * self.m * self.m)
        self.features = nn.Sequential(
            deconv(16 * ch, 16 * ch, 2),
            deconv(16 * ch, 8 * ch, 2),
            deconv(8 * ch, 4 * ch, 2),
            deconv(4 * ch, 2 * ch, 2),
            deconv(2 * ch, 3, act=False)
        )
        self.tan = nn.Tanh()
        # if init:
        #     init_model(self)

    def forward(self, z, get_activations=False, get_before_act_func=True):
        x = torch.nn.functional.relu(self.encode(z))
        x = x.view(x.size()[0], self.ch * 16, self.m, self.m)
        x = self.features(x)
        return self.tan(x)


class Discriminator(nn.Module):
    def __init__(self, latent_dim=128, prior=False, n_classes=4, init=True):
        super(Discriminator, self).__init__()
        if latent_dim > 512:
            self.model = nn.Sequential(
                nn.Linear(latent_dim + (n_classes if prior else 0), latent_dim // 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(latent_dim // 2, latent_dim // 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(latent_dim // 4, latent_dim // 6),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(latent_dim // 6, 1),
                nn.Sigmoid(),
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(latent_dim + (n_classes if prior else 0), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

    def forward(self, z):
        validity = self.model(z)
        return validity
