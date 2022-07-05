import torch
from torch import nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# ResBlockDecoder and make_layer function inspired and modified by https://github.com/julianstastny/VAE-ResNet18-PyTorch
class MyConvTranspose2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.conv(x)
        return x


class ResBlockDecoder(nn.Module):
    def __init__(self, inplanes, stride):
        super().__init__()
        self.planes = int(inplanes/stride)
        self.convBlock2_de = nn.Sequential(nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                                           nn.BatchNorm2d(inplanes),
                                           nn.ReLU())

        if stride == 1:
            self.convBlock1_de = nn.Sequential(nn.Conv2d(inplanes, self.planes, kernel_size=3, stride=1, padding=1, bias=False),
                                               nn.BatchNorm2d(self.planes))
            self.shortcut = nn.Sequential()
        # if stride > 1, that means decoder need to do upsampling
        else:
            self.convBlock1_de = nn.Sequential(MyConvTranspose2d(inplanes, self.planes, 3, scale_factor=stride),
                                               nn.BatchNorm2d(self.planes))
            # nn.ConvTranspose2d is a little bit difficult to manipulate the output size
            # n.ConvTranspose2d(inplanes, self.planes, kernel_size=3, stride=stride, output_padding=1)
            self.shortcut = nn.Sequential(MyConvTranspose2d(inplanes, self.planes, 3, scale_factor=stride),
                                          nn.BatchNorm2d(self.planes))

    def forward(self, x):
        output = self.convBlock2_de(x)
        output = self.convBlock1_de(output)
        sc = self.shortcut(x)
        output += self.shortcut(x)
        return torch.relu(output)

def make_layer(ResBlockDecoder, inplanes, stride):
    strides = [stride] + [1]
    layers = []
    for stride in reversed(strides):
        layers += [ResBlockDecoder(inplanes, stride)]
    return nn.Sequential(*layers)

class VAEencoder(nn.Module):
    def __init__(self, model, hidden_dim=256):
        super().__init__()
        self.extractor = model
        self.fc_mu = nn.Linear(512, hidden_dim)
        self.fc_sigma = nn.Linear(512, hidden_dim)

    def forward(self, x):
        output = self.extractor(x)
        output = output.view(x.size(0), -1)
        mu = self.fc_mu(output)
        logvar = self.fc_sigma(output)
        return mu, logvar


class VAEdecoder(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=512)
        self.layer4 = make_layer(ResBlockDecoder, 512, stride=2)    # bsz 256 2 2
        self.layer3 = make_layer(ResBlockDecoder, 256, stride=2)
        self.layer2 = make_layer(ResBlockDecoder, 128, stride=2)
        self.layer1 = make_layer(ResBlockDecoder, 64, stride=1)     # bsz 64 8 8
        self.conv_de = MyConvTranspose2d(64, 3, kernel_size=3, scale_factor=2)

    def forward(self,x):
        output = self.linear(x)
        output = output.view(x.size(0), 512, 1, 1)
        output = F.interpolate(output, scale_factor=4)
        output = self.layer4(output)
        output = self.layer3(output)
        output = self.layer2(output)
        output = self.layer1(output)
        output = torch.sigmoid(self.conv_de(output))

        #output = output.view(x.size(0), 3, 64, 64)
        return output

class ResVAE(nn.Module):
    def __init__(self, pre_trained=False, hidden_dim=256):
        super().__init__()
        self.pre_trained = pre_trained
        self.hidden_dim = hidden_dim
        self.encoder = self._make_encoder()
        self.decoder = self._make_decoder()

    def _make_encoder(self):
        model = models.resnet18(pretrained=self.pre_trained)
        model.fc = nn.Identity()
        encoder = VAEencoder(model)
        return encoder

    def _make_decoder(self):
        decoder = VAEdecoder(self.hidden_dim)
        return decoder

    def reparameterize(self, mu, log_var):
        """ Reparametrization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, x):
        #encoder_output = self.encoder(x)
        #decoder_output = self.decoder(encoder_output)
        mu, log_var = self.encoder(x)

        # reparametrization trick
        z = self.reparameterize(mu, log_var)

        # decoding
        x_hat = self.decoder(z)

        return x_hat, (z, mu, log_var)


if __name__ == '__main__':
    model = ResVAE()
    # # input = torch.randn(8, 3, 28, 28)
    # # down = nn.Conv2d(3, 64, kernel_size=3, stride=2)
    # # up = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, output_padding=1)
    # # x = down(input)
    # # print(x.shape)
    # # output = up(x)
    # # print(output.shape)



