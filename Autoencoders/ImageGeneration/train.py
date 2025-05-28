import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import Autoencoder

transform = transforms.Compose([
    transforms.Grayscale(),  # if needed
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

dataset = ImageFolder('data/', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Autoencoder(latent_dim=64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(20):
    for imgs, _ in dataloader:
        output = model(imgs)
        loss = criterion(output, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

torch.save(model.state_dict(), 'checkpoints/autoencoder.pth')
