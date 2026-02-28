import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ConvNextEncoder, ConvNextDecoder
from .iaf import IAFModel
from ..config.models import ConvVAEConfig


class ConvVAE(nn.Module):
    def __init__(self, config: ConvVAEConfig):
        super().__init__()

        in_channels = config.in_channels

        self.encoder = ConvNextEncoder(
            in_channels=in_channels,
            depths=config.depths,
            dims=config.dims,
            ln_eps=config.ln_eps
        )

        self.decoder = ConvNextDecoder(
            latent_dim=config.latent_dim,
            out_channels=in_channels,
            depths=config.depths[::-1],
            dims=config.dims[::-1],
            ln_eps=config.ln_eps
        )

        self.mu_proj = nn.Conv2d(
            config.dims[-1],
            config.latent_dim,
            kernel_size=1,
            padding='same'
        )
        self.logvar_proj = nn.Conv2d(
            config.dims[-1],
            config.latent_dim,
            kernel_size=1,
            padding='same'
        )
        self.h_proj = nn.Conv2d(
            config.dims[-1],
            config.latent_dim,
            kernel_size=1,
            padding='same'
        )

        self.n_blocks = len(config.depths)
        self.f0 = config.dims[0]
        self.image_size = config.image_size

        self.latent_image_size = self.image_size // (2 ** (self.n_blocks-1))
        self.latent_channels = config.dims[-1]
        self.latent_dim = config.latent_dim

        self.iaf = config.iaf
        if self.iaf:
            self.iaf_model = IAFModel(
                latent_dim=config.latent_dim,
                latent_image_size=self.latent_image_size,
                iaf_steps=config.iaf_timesteps,
                ln_eps=config.ln_eps,
                n_blocks=config.iaf_n_blocks
            )
            self.iaf_timesteps = config.iaf_timesteps

        nn.init.zeros_(self.mu_proj.weight)
        nn.init.zeros_(self.mu_proj.bias)
        nn.init.zeros_(self.logvar_proj.weight)
        nn.init.zeros_(self.logvar_proj.bias)
        nn.init.zeros_(self.h_proj.weight)
        nn.init.zeros_(self.h_proj.bias)

    def encode(self, x):
        x = self.encoder(x)

        mu = self.mu_proj(x)
        log_var = self.logvar_proj(x)
        h = self.h_proj(x)

        return mu, log_var, h

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): An input tensor of shape BxCx64x64

        Returns:
            _type_: _description_
        """
        mu, log_var, h = self.encode(x)
        z, eps = self.reparameterize(mu, log_var)

        if self.iaf:
            B, C, W, H = z.shape
            log_vars = [log_var]
            zs = [z]
            for timestep in range(self.iaf_timesteps):
                t = torch.full((B, 1, W, H), timestep/self.iaf_timesteps, device=z.device, dtype=z.dtype)
                m, s = self.iaf_model(z, h, t, *zs)
                var = F.sigmoid(s)
                z = var * z + (1 - var) * m
                log_var = torch.log(var + 1e-8)
                log_vars.append(log_var)
                zs.append(z)
            reconstruction = self.decode(z)
            return reconstruction, z, eps, log_vars

        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

    @torch.no_grad()
    def generate(self, batch_size=1, device='cuda'):
        """Generate new images from random latent vectors.

        Args:
            batch_size (int): Number of images to generate
            device (str): Device to generate images on

        Returns:
            torch.Tensor: Generated images of shape (batch_size, channels, height, width)
        """
        latent_shape = (self.latent_dim, self.latent_image_size, self.latent_image_size)
        z = torch.randn(batch_size, *latent_shape, device=device)
        return torch.sigmoid(self.decode(z))
