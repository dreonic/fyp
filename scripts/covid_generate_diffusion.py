import sys
import os
import argparse
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.nn import functional as F
import tqdm
import matplotlib.pyplot as plt

# Import model architecture from covid-diffusion
# We need to import all the model classes
import importlib.util
spec = importlib.util.spec_from_file_location("covid_diffusion", os.path.join(os.path.dirname(__file__), "covid-diffusion.py"))
covid_diffusion = importlib.util.module_from_spec(spec)

# Set global device before executing the module (required for PositionalEncoding)
# This is a workaround since covid-diffusion.py uses a global 'device' variable
import builtins
_original_device = None

def load_model_with_device(device_obj):
    """Load covid_diffusion module with the correct device context"""
    # Temporarily inject 'device' into the module's namespace
    global _original_device
    _original_device = getattr(builtins, 'device', None)
    
    # Execute the module with device available
    covid_diffusion.__dict__['device'] = device_obj
    spec.loader.exec_module(covid_diffusion)
    
    return covid_diffusion.UNet

def run_inference(unet, num_images, T, alphas, alpha_bars, device, image_size=128):
    """
    Generate images using the trained diffusion model
    
    Parameters
    ----------
    unet : UNet model
        Trained diffusion model
    num_images : int
        Number of images to generate
    T : int
        Number of diffusion timesteps
    alphas : torch.Tensor
        Alpha values for diffusion
    alpha_bars : torch.Tensor
        Cumulative alpha values
    device : torch.device
        Device to run on
    image_size : int
        Size of generated images
    """
    print(f"\nGenerating {num_images} images...")
    print("This may take a few minutes depending on the number of images and timesteps.")
    
    unet.eval()
    # 0. generate sigma_t
    alpha_bars_prev = torch.cat((torch.ones(1).to(device), alpha_bars[:-1]))
    sigma_t_squared = (1.0 - alphas) * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    sigma_t = torch.sqrt(sigma_t_squared)
    
    # 1. make white noise with correct dimensions for grayscale images
    x = torch.randn(num_images, 1, image_size, image_size).to(device)
    
    # 2. loop through diffusion steps
    with torch.no_grad():
        for t in tqdm.tqdm(reversed(range(T)), total=T, desc="Denoising"):
            if t > 0:
                z = torch.randn_like(x).to(device)
            else:
                z = torch.zeros_like(x).to(device)
            t_batch = (torch.tensor(t).to(device)).repeat(num_images)
            epsilon = unet(x, t_batch)
            x = (1.0 / torch.sqrt(alphas[t])).float() * (x - ((1.0 - alphas[t]) / torch.sqrt(1.0 - alpha_bars[t])).float() * epsilon) + \
                sigma_t[t].float() * z

    # reshape to channels-last : (N,C,H,W) --> (N,H,W,C)
    x = x.permute(0, 2, 3, 1)

    # clip
    x = torch.clamp(x, min=0.0, max=1.0)

    return x

def main(checkpoint_path, output_dir, num_images=10, image_size=128, device_obj=None):
    """
    Load a trained diffusion model and generate synthetic images
    
    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint (.pt file)
    output_dir : str
        Directory to save generated images
    num_images : int
        Number of images to generate
    image_size : int
        Size of images (must match training size)
    device_obj : torch.device
        Device to run on
    """
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("COVID-19 Diffusion Model - Image Generation")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of images: {num_images}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Device: {device_obj}")
    print("="*60)
    
    # Load UNet model class with proper device context
    print("\nLoading model architecture...")
    UNet = load_model_with_device(device_obj)
    
    # Initialize model
    print("Initializing model...")
    unet = UNet(
        source_channel=1,  # Grayscale images
        unet_base_channel=128,
        num_norm_groups=32,
    ).to(device_obj)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)
    unet.load_state_dict(checkpoint)
    print("Checkpoint loaded successfully!")
    
    # Initialize diffusion parameters (same as training)
    T = 1000
    alphas = torch.linspace(start=0.9999, end=0.98, steps=T, dtype=torch.float64).to(device_obj)
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    # Generate images
    generated_images = run_inference(unet, num_images, T, alphas, alpha_bars, device_obj, image_size)
    
    # Save generated images
    print(f"\nSaving generated images to {output_dir}/")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i in range(num_images):
        img = generated_images[i].cpu().numpy()
        filename = f"generated_{timestamp}_{i+1}.png"
        filepath = os.path.join(output_dir, filename)
        plt.imsave(filepath, img.squeeze(), cmap='gray')
        print(f"  Saved: {filename}")
    
    print(f"\n✓ Successfully generated {num_images} images!")
    print(f"✓ Images saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate synthetic COVID-19 CT scan images using a trained diffusion model'
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint file (.pt)')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./generated_diffusion_images',
                        help='Directory to save generated images (default: ./generated_diffusion_images)')
    parser.add_argument('--num_images', type=int, default=10,
                        help='Number of images to generate (default: 10)')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Size of generated images - must match training size (default: 128)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use (default: 0)')
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        print("WARNING: Running on CPU. This will be very slow!")
    
    main(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_images=args.num_images,
        image_size=args.image_size,
        device_obj=device,
    )

    