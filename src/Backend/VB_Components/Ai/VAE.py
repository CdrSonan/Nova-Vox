import torch
import torch.nn as nn

import global_consts

class VAE(nn.Module):
    def __init__(self, device, latent_dim=128, learningRate:float = 0.0001):
        super().__init__()

        input_dim = global_consts.halfTripleBatchSize + global_consts.halfHarms + 1

        self.device = device
        self.learningRate = learningRate

        # encoder, decoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2, device = self.device),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2, device = self.device),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2, device = self.device),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim, device = self.device),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2, device = self.device),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2, device = self.device),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2, device = self.device),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, input_dim, device = self.device),
            nn.ReLU()
        )
    
        self.mean = nn.Linear(latent_dim, latent_dim, device = self.device)
        self.logvar = nn.Linear(latent_dim, latent_dim, device = self.device)

    def encode(self, x):
        encoded = self.encoder(x)
        mean = self.mean(encoded)
        logvar = self.logvar(encoded)
        return mean, logvar
    
    def encode_infer(self, x):
        mean, logvar = self.encode(x)
        return mean

    def decode(self, x):
        decoded = self.decoder(x)
        return decoded
    
    def reparametrize(self, mean, var):
        eps = torch.randn_like(var, device = self.device)
        return eps * var + mean
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparametrize(mean, torch.exp(0.5 * logvar))
        decoded = self.decode(z)
        return decoded, mean, logvar
    
    def loss(self, x, decoded, mean, logvar):
        reconstruction_loss = torch.mean(torch.abs(decoded - x))
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return reconstruction_loss + kl_loss

    def training_step(self, x):
        decoded, mean, logvar = self.forward(x)
        loss = self.loss(x, decoded, mean, logvar)
        return loss
