import torch
from vae import VAE
import matplotlib.pyplot as plt

# Configuration
width = 30
height = 60
grayscale = True
latent_dim = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = VAE(width=width, height=height, grayscale=grayscale, latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load("vae_custom.pth", map_location=device))
model.eval()

# Generate 1 sample
with torch.no_grad():
    z = torch.randn(1, latent_dim).to(device)
    generated = model.decode(z).cpu()

# Convert to numpy for display
img = generated[0]

if grayscale:
    plt.imshow(img[0], cmap='gray')  # shape [1, H, W] → [H, W]
else:
    img = img.permute(1, 2, 0)  # shape [3, H, W] → [H, W, 3]
    plt.imshow(img)

plt.axis('off')
plt.title("Generated Image")
plt.show()
