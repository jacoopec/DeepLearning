import torch
from model import Autoencoder
import matplotlib.pyplot as plt

model = Autoencoder(latent_dim=64)
model.load_state_dict(torch.load('checkpoints/autoencoder.pth'))
model.eval()

# Generate random samples
with torch.no_grad():
    latent = torch.randn(16, 64)  # 16 new samples
    generated = model.decoder(latent)

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated[i][0].numpy(), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
