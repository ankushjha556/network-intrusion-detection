import torch
import torch.nn as nn

class NetworkAutoencoder(nn.Module):
    def __init__(self, input_dim=40, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.BatchNorm1d(32),
            nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 24), nn.BatchNorm1d(24), nn.GELU(),
            nn.Linear(24, 16), nn.BatchNorm1d(16), nn.GELU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.BatchNorm1d(16), nn.GELU(),
            nn.Linear(16, 24), nn.BatchNorm1d(24), nn.GELU(),
            nn.Linear(24, 32), nn.BatchNorm1d(32), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(32, input_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
    def reconstruction_error(self, x):
        with torch.no_grad():
            recon, _ = self.forward(x)
            return torch.mean((x - recon)**2, dim=1)
