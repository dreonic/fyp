
from matplotlib import transforms
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_data_loader(train_dir, image_size=64, batch_size=16):
    """
    Create PyTorch data loader with augmentation for training
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    print(f"Data loader created.")
    print(f"Number of training samples: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Classes: {dataset.classes}")
    
    return loader
