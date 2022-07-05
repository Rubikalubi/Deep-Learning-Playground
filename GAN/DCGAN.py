import torch
import torch.nn as nn


torch.manual_seed(0)

# Implement deep convolutional GANs from https://arxiv.org/pdf/1511.06434.pdf
class DCGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.ConvTranspose2d(100, 512, 4, 1, 0),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())
        self.conv3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU())
        self.conv4 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1),
                                   nn.BatchNorm2d(64),
                                    nn.ReLU())
        self.conv5 =  nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1),
                                   nn.Tanh())

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1) # BSZ, hidden_dim, 1, 1
        output = self.conv1(x)            # BSZ, 1024, 3, 3
        output = self.conv2(output)       # BSZ, 512, 7, 7
        output = self.conv3(output)       # BSZ, 256, 14, 14
        output = self.conv4(output)       # BSZ, 1, 28, 28
        output = self.conv5(output)
        return output

class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        #just some conv layers
        self.layers = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(64, 128, 4, 2, 1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(128, 256, 4, 2, 1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(256, 512, 4, 2, 1),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.2),
                                    nn.Conv2d(512, 1, 4, 1, 0)
                                    )

    def forward(self, x):
        output = self.layers(x)
        output = output.view(x.shape[0], -1)
        return output

if __name__ == '__main__':
    a = torch.ones((64,1), device="cuda")
    print(a.shape)