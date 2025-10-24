import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
from torch.nn import functional as F

latent_dim = 32

def generate_images(decoder_net, device="cuda", num_images=5):
    latent_samples = torch.randn(num_images, latent_dim).to(device)
    with torch.no_grad():
        imgs = decoder_net(latent_samples)
    for i in imgs:
        plt.imshow(i[0].cpu().numpy(), cmap='gray')
        plt.show()

def create_data_loader(train_dir, image_size=64, batch_size=16):
    """
    Create PyTorch data loader with augmentation for training
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    
    # Create data loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print(f"Data loader created.")
    print(f"Number of training samples: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Classes: {dataset.classes}")
    
    return loader

class EncoderNet(nn.Module):
    def __init__(self, latent_dim, image_size=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # Calculate flattened dimension after convolutions
        # After 3 stride-2 convs: image_size / (2^3) = image_size / 8
        self.flat_dim = 128 * (image_size // 8) * (image_size // 8)
        
        self.fc_mean = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar
    
class DecoderNet(nn.Module):
    def __init__(self, latent_dim, image_size=64):
        super().__init__()
        self.image_size = image_size
        # Calculate the starting dimension for decoder
        self.start_dim = image_size // 8
        self.projection = nn.Linear(latent_dim, 128 * self.start_dim * self.start_dim)
        
        self.convtrans1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.convtrans2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.convtrans3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.projection(x)
        x = torch.reshape(x, (-1, 128, self.start_dim, self.start_dim))
        x = F.relu(self.convtrans1(x))
        x = F.relu(self.convtrans2(x))
        x = torch.sigmoid(self.convtrans3(x))
        return x
    
def kl(mean, logvar):
    elements = logvar + 1.0 - mean**2 - torch.exp(logvar)
    return torch.sum(elements, dim=-1) * (-0.5)

# expand tensor, such as : [batch_size, latent_dim] --> [batch_size * num_sampling, latent_dim]
def expand_for_sampling(x, num_sampling):
    latent_dim = x.shape[1]

    x_sampling = x.unsqueeze(dim=1)
    x_sampling = x_sampling.expand(-1, num_sampling, -1)
    return torch.reshape(x_sampling, (-1, latent_dim))

def reparameter_sampling(mean, logvar, num_sampling, device='cuda'):
    # expand mean : [batch_size, latent_dim] --> [batch_size * num_sampling, latent_dim]
    mean_sampling = torch.repeat_interleave(mean, num_sampling, dim=0)

    # expand logvar : [batch_size, latent_dim] --> [batch_size * num_sampling, latent_dim]
    logvar_sampling = torch.repeat_interleave(logvar, num_sampling, dim=0)

    # get epsilon
    epsilon_sampling = torch.randn_like(mean_sampling).to(device)

    return torch.mul(torch.exp(logvar_sampling*0.5), epsilon_sampling) + mean_sampling

def main(data_dir, output_dir, num_images=10):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(f"Using device: {device}")

    image_size = 64  # Image size for the model
    batch_size = 16
    
    # Create data loader
    loader = create_data_loader(data_dir, image_size=image_size, batch_size=batch_size)
    
    #
    # Generate a model for encoder distribution
    #
    encoder_net = EncoderNet(latent_dim=latent_dim, image_size=image_size).to(device)

    #
    # Generate a decoder model
    #
    decoder_net = DecoderNet(latent_dim=latent_dim, image_size=image_size).to(device)

    num_sampling = 5     # the number of sampling
    num_epochs = 20

    opt = torch.optim.AdamW(list(encoder_net.parameters()) + list(decoder_net.parameters()), lr=0.001)

    loss_records = []
    for epoch_idx in range(num_epochs):
        for batch_idx, (data, _) in enumerate(loader):
            x = data.to(device)

            opt.zero_grad()

            # get mean and logvar
            z_mean, z_logvar = encoder_net(x)
        
            # get KL-div (KL loss)
            kl_loss_batch = kl(z_mean, z_logvar) # shape: [batch_size,]
            kl_loss = torch.sum(kl_loss_batch)
        
            # sampling by reparameterization
            z_samples = reparameter_sampling(z_mean, z_logvar, num_sampling, device=device)
        
            # reconstruct x by decoder
            decoded_x_samples = decoder_net(z_samples)
            decoded_x_samples = torch.reshape(decoded_x_samples, (-1, image_size*image_size))
        
            # get BCE loss between the input x and the reconstructed x
            x_samples = torch.repeat_interleave(x, num_sampling, dim=0)  # expand to shape: [batch_size * num_sampling, 1, H, W]
            x_samples = torch.reshape(x_samples, (-1, image_size*image_size))
            pvar = torch.ones_like(x_samples).to(device) * 0.25  # sharp fit (p(x)_sigma=0.5)
            # pvar = torch.ones_like(x_samples).to(device) * 1.00  # loose fit (p(x)_sigma=1.0)
            reconstruction_loss = F.gaussian_nll_loss(
                decoded_x_samples,
                x_samples,
                var=pvar,
                reduction="sum"
            )
            reconstruction_loss /= num_sampling
        
            # optimize parameters
            total_loss = reconstruction_loss + kl_loss
            total_loss.backward()
            opt.step()
        
            # log
            loss_records.append(total_loss.item())
            print("epoch{} (iter{}) - loss {:5.4f}".format(epoch_idx, batch_idx, total_loss.item()), end="\r")

        epoch_loss_all = loss_records[-(batch_idx+1):]
        average_loss = sum(epoch_loss_all)/len(epoch_loss_all)
        print("epoch{} (iter{}) - loss {:5.4f}".format(epoch_idx, batch_idx, average_loss))

    print("Done training!")

    # Generate synthetic images
    print(f"\nGenerating {num_images} synthetic images...")
    generate_images(decoder_net, device=device, num_images=num_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train COVID-19 CT Scan VAE to Generate Synthetic Images'
    )

    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory containing Covid/Healthy/Others folders')

    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./generated_images',
                        help='Directory to save generated images')
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_images=10
    )
        