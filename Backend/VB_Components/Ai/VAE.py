import torch
import torch.nn as nn

import global_consts

class VAE(nn.Module):
    def __init__(self, device, latent_dim=128, learningRate:float = 0.0001):
        super().__init__()

        encoder_dim = global_consts.halfTripleBatchSize + global_consts.halfHarms

        self.device = device
        self.learningRate = learningRate

        # encoder, decoder
        self.encoder = nn.Sequential(
            nn.Linear(encoder_dim, latent_dim * 2, device = self.device),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim * 2, latent_dim, device = self.device),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2, device = self.device),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_dim * 2, encoder_dim, device = self.device),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # distribution parameters
        self.fc_mu = nn.Linear(encoder_dim, latent_dim, device = self.device)
        self.fc_var = nn.Linear(encoder_dim, latent_dim, device = self.device)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.tensor([0.0,], device = self.device))

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, x):

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        return elbo