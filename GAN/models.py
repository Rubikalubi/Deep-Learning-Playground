from turtle import forward
import torch
import torch.nn as nn
# Experiment 1:
# Implement a fully connected conditional GAN (https://arxiv.org/abs/1411.1784)
# Train the model for conditional generation on the Fashion MNIST dataset
# Requirements:
# Use Tensorboard, WandDB or some other experiment tracker
# Show the capabilities of the model to generate data based on given label


class ConditionalGenerator(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(ConditionalGenerator, self).__init__()

        self.input_dim = input_dim + 10  # for the 10 classes
        self.output_dim = output_dim

        # we need the embedding because the labels are long tensors and we need to map them to float tensors
        self.embedding = nn.Embedding(10, 10)

        self.hidden_1 = nn.Sequential(
            nn.Linear(self.input_dim, 256), nn.LeakyReLU())

        self.hidden_2 = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU())

        self.hidden_3 = nn.Sequential(nn.Linear(512, 1024), nn.LeakyReLU())

        self.hidden_4 = nn.Sequential(
            nn.Linear(1024, self.output_dim), nn.Tanh())

    def forward(self, x, labels):
        """forward function

        Args:
            x (torch tensor): batch of inputs
            labels (torch tensor): batch of labels
        """

        embeddings = self.embedding(labels)

        # concatenate latent dimension and embedding at dimension 1
        x = torch.cat([x, embeddings], 1)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        x = self.hidden_4(x)
        return x


class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(ConditionalDiscriminator, self).__init__()

        self.input_dim = input_dim + 10  # for the 10 classes
        self.output_dim = output_dim

        # we need the embedding because the labels are long tensors and we need to map them to float tensors
        self.embedding = nn.Embedding(10, 10)

        self.hidden_1 = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.ReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hidden_4 = nn.Sequential(
            nn.Linear(256, self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        embeddings = self.embedding(labels)
        x = torch.cat([x, embeddings], 1)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        x = self.hidden_4(x)
        return x
