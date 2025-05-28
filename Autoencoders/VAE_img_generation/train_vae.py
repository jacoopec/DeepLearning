import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision import transforms
from vae import VAE
import os
from PIL import Image

# Configuration
width = 30
height = 60
grayscale = True
latent_dim = 30
batch_size = 8
epochs = 1000
lr = 1e-3
image_dir = "imgs/train"  # must contain subfolders, e.g. imgs/train/dummy/image.png

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations
transform_list = [transforms.Resize((width, height))]
if grayscale:
    transform_list.append(transforms.Grayscale(num_output_channels=1))
transform_list.append(transforms.ToTensor())
transform = transforms.Compose(transform_list)

# Dataset and DataLoader
dataset = ImageFolder(root=image_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model and optimizer
model = VAE(width=width, height=height, grayscale=grayscale, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# VAE loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for x, _ in train_loader:
        x = x.to(device)
        recon, mu, logvar = model(x)
        loss = vae_loss(recon, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader.dataset):.4f}")

# Save the model
torch.save(model.state_dict(), "vae_custom.pth")
