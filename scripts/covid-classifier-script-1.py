import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
import argparse

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import matplotlib.image as mpimg

def benchmark(model, train_loader, val_loader, test_loader, device):
    # Calculate accuracy for all datasets
    from sklearn import metrics

    # Training set accuracy
    train_gts, train_preds, _, _ = infer(train_loader, model, device)
    train_accuracy = metrics.accuracy_score(train_gts, train_preds)

    # Validation set accuracy
    val_gts, val_preds, _, _ = infer(val_loader, model, device)
    val_accuracy = metrics.accuracy_score(val_gts, val_preds)

    # Test set accuracy
    test_gts, test_preds, _, _ = infer(test_loader, model, device)
    test_accuracy = metrics.accuracy_score(test_gts, test_preds)

    print("=" * 50)
    print("CLASSIFIER ACCURACY SUMMARY")
    print("=" * 50)
    print(f"Training Set Accuracy:   {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Validation Set Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"Test Set Accuracy:       {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("=" * 50)

def infer(loader, model, device):
    gts = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            out = model(images.to(device))
            gts.extend(labels.tolist())
            predictions.extend(out.argmax(1).tolist())

    from sklearn import metrics
    cm = metrics.confusion_matrix(gts, predictions, normalize='true')
    return gts, predictions, np.unique(gts, return_counts=True), cm


class dataset(Dataset):
    def __init__(self, df, class_dict):
        super().__init__()
        self.df = df
        self.class_dict = class_dict

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        image_path = row['Image path']
        target = self.class_dict[row['target']]

        image = mpimg.imread(image_path)[:,:,:3]
        image = image / 255.0 if image.max() > 1.0 else image
        return torch.Tensor(image).reshape(3, *image.shape[:2]), torch.Tensor([target])
    
def collate_fn(batch):
    dims = np.array([tuple(x[0].shape) for x in batch])
    max_dims = dims.max(0)

    out = torch.zeros(len(batch), *max_dims)
    labels = torch.zeros(len(batch))
    for i, (image, label) in enumerate(batch):
        out[i, :, :image.shape[1], :image.shape[2]] = image
        labels[i] = label
        
    return out, labels.long()


def main(batch_size=16, num_workers=0, epochs=10, lr_power=4):
    # This Python 3 environment comes with many helpful analytics libraries installed
    # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
    # For example, here's several helpful packages to load

    # Calculate learning rate from power: lr = 10^(-lr_power)
    lr = 10 ** (-lr_power)

    root_dir = './input/a-covid-multiclass-dataset-of-ct-scans/New_Data_CoV2'
    root_path = Path(root_dir)
    
    # Check if data directory exists
    if not root_path.exists():
        raise FileNotFoundError(f"Data directory not found: {root_path.absolute()}\n"
                                f"Please ensure the dataset is downloaded and the path is correct.")
    
    covid_path = root_path / 'Covid'
    healthy_path = root_path / 'Healthy'
    others_path = root_path / 'Others'

    images = list(root_path.rglob('*.png'))
    
    if len(images) == 0:
        raise ValueError(f"No PNG images found in {root_path.absolute()}\n"
                        f"Please check the data directory structure.")
    
    print(f"Found {len(images)} images in dataset")
    
    patients = [p.parts[-2] for p in images]
    target = [p.parts[-3] for p in images]

    df = pd.DataFrame(np.array([images, patients, target]).T, columns=['Image path', 'Patient', 'target'])
    
    print(f"\nInitial dataset distribution:")
    print(df.groupby('target').size())

    # Balance dataset

    oversample = False
    undersample = True

    if oversample:
        healthy_rows_idxs = df[df['target'] == 'Healthy'].index.values.tolist()
        non_healthy_rows_idxs = df.index.values.tolist() 
        df = df.iloc[healthy_rows_idxs*2 + non_healthy_rows_idxs]
        
    if undersample:
        # Check how many Covid samples exist
        covid_samples = df[df['target'] == 'Covid'].index.values
        
        if len(covid_samples) == 0:
            print("WARNING: No Covid samples found in dataset. Skipping undersampling.")
        else:
            # Use the minimum of 750 or available Covid samples
            n_samples = min(750, len(covid_samples))
            print(f"Undersampling: Using {n_samples} Covid samples")
            
            covid_selected = np.random.choice(covid_samples, n_samples, replace=False).tolist()
            non_covid = df[df['target'] != 'Covid'].index.values.tolist()
            
            df = df.iloc[non_covid + covid_selected]
    
    print(f"\nFinal dataset distribution:")
    print(df.groupby('target').size())
    print(f"Total samples: {len(df)}\n")

    # Split dataset

    from sklearn.model_selection import train_test_split
    train_patients, testval_patients = train_test_split(
        df.Patient.unique(),
        train_size = 0.8
    )

    test_patients, val_patients = train_test_split(
        testval_patients,
        train_size = 0.5
    )

    train_df = df[df['Patient'].isin(train_patients)]
    test_df = df[df['Patient'].isin(test_patients)]
    val_df = df[df['Patient'].isin(val_patients)]

    # Create dataloaders

    model = nn.Sequential(
        nn.Conv2d(3, 4, 3),
        nn.BatchNorm2d(4),
        nn.Conv2d(4, 16, 3),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(16, 32, 3),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 32, 3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, 3)
    )


    class_dict = {'Healthy': 0, 'Covid': 1, 'Others': 2}

    # Hyperparameters (passed as function parameters)
    print(f"Training with: batch_size={batch_size}, num_workers={num_workers}, epochs={epochs}, lr=1e-{lr_power} ({lr})")
    
    # Check if CUDA is available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        # Clear any existing CUDA cache
        torch.cuda.empty_cache()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
    else:
        print("CUDA not available, using CPU")
    
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_set = dataset(train_df, class_dict)
    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)

    val_set = dataset(val_df, class_dict)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)

    test_set = dataset(test_df, class_dict)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)

    losses = []

    for epoch in range(epochs):
        epoch_train_loss = 0
        model.train()
        for images, labels in train_loader:
            out = model(images.to(device))
            loss = criterion(out, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() / len(train_loader)
                    
        epoch_validation_loss = 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                out = model(images.to(device))
                loss = criterion(out, labels.to(device))

                epoch_validation_loss += loss.item() / len(val_loader)
        out = infer(val_loader, model, device)
        print(f'Epoch {epoch}:\t train loss {epoch_train_loss:0.4e}\t val loss {epoch_validation_loss:0.4e}')
        print(out[-1])
        losses.append([epoch_train_loss, epoch_validation_loss])

    benchmark(model, train_loader, val_loader, test_loader, device)


    if device == 'cuda':
        print(f"Final GPU memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"Final GPU memory reserved: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
        torch.cuda.empty_cache()
        print("Cleared GPU cache")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train COVID-19 CT Scan Classifier')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training (default: 16)')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Number of data loading workers (default: 0)')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--lr_power', type=int, default=4, 
                        help='Learning rate as 10^(-lr_power), e.g., 4 means 1e-4 (default: 4)')
    
    args = parser.parse_args()
    
    main(batch_size=args.batch_size, 
         num_workers=args.num_workers, 
         epochs=args.epochs, 
         lr_power=args.lr_power)