import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, width, height, grayscale=True, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        self.width = width
        self.height = height
        self.channels = 1 if grayscale else 3
        self.input_dim = self.channels * self.width * self.height

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 512),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z).view(-1, self.channels, self.width, self.height)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
