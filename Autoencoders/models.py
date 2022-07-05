from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import torch.nn.functional as F

# DISCLAIMER:

# We took this code from  https://debuggercafe.com/generating-fictional-celebrity-faces-using-convolutional-variational-autoencoder-and-pytorch/ and modified it to be able to use images of size 128x128 and also be able to generate images from random noise


# define a Conv VAE
class ConvVAE(nn.Module):
    def __init__(self, init_channels=64, image_channels=3, latent_dim=128):
        super(ConvVAE, self).__init__()

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels,
            kernel_size=4, stride=2, padding=2
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2,
            kernel_size=4, stride=2, padding=2
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4,
            kernel_size=4, stride=2, padding=2
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=init_channels*8,
            kernel_size=4, stride=2, padding=2
        )
        self.enc5 = nn.Conv2d(
            in_channels=init_channels*8, out_channels=1024,
            kernel_size=4, stride=2, padding=2
        )
        self.fc1 = nn.Linear(1024, 2048)
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_log_var = nn.Linear(2048, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 1024)
        # decoder
        self.dec1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=init_channels*8,
            kernel_size=3, stride=2
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4,
            kernel_size=3, stride=2
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2,
            kernel_size=3, stride=2
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=init_channels,
            kernel_size=3, stride=2
        )
        self.dec5 = nn.ConvTranspose2d(
            in_channels=init_channels, out_channels=image_channels,
            kernel_size=3, stride=2
        )
        self.dec6 = nn.ConvTranspose2d(
            in_channels=image_channels, out_channels=image_channels,
            kernel_size=4, stride=2
        )

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling
        return sample

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = F.relu(self.enc5(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        return mu, log_var

    def latent_vector(self, mu, log_var):
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, 1024, 1, 1)
        return z

    def decoder(self, z):
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        reconstruction = torch.sigmoid(self.dec6(x))
        return reconstruction

    def forward(self, x):
        # encoding
        mu, log_var = self.encode(x)

        # get the latent vector through reparameterization
        # z = self.reparameterize(mu, log_var)
        # z = self.fc2(z)
        # z = z.view(-1, 1024, 1, 1)
        z = self.latent_vector(mu, log_var)
        reconstruction = self.decoder(z)

        # decoding
        # x = F.relu(self.dec1(z))
        # x = F.relu(self.dec2(x))
        # x = F.relu(self.dec3(x))
        # x = F.relu(self.dec4(x))
        # x = F.relu(self.dec5(x))
        # reconstruction = torch.sigmoid(self.dec6(x))
        return reconstruction, mu, log_var
