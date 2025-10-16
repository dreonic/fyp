"""
COVID-19 CT Scan Multiclass Classifier using Transfer Learning (EfficientNetB2)

This script implements a deep learning model for classifying COVID-19 CT scans
into three categories: Covid, Healthy, and Others.

"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# EfficientNet
try:
    from efficientnet.tfkeras import EfficientNetB2
except ImportError:
    print("ERROR: efficientnet not installed. Please run: pip install efficientnet")
    sys.exit(1)


def create_split_directories(output_dir):
    """Create train/val/test directory structure"""
    for split in ['train', 'val', 'test']:
        for class_name in ['Covid', 'Healthy', 'Others']:
            dir_path = os.path.join(output_dir, split, class_name)
            os.makedirs(dir_path, exist_ok=True)


def split_dataset(base_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split dataset into train/val/test directories using patient-level splitting
    """
    print(f"\nSplitting dataset: {train_ratio}/{val_ratio}/{test_ratio}")
    
    # Load all images and create dataframe
    root_path = Path(base_dir)
    images = list(root_path.rglob('*.png'))
    
    if len(images) == 0:
        raise ValueError(f"No PNG images found in {base_dir}")
    
    print(f"Found {len(images)} images")
    
    patients = [p.parts[-2] for p in images]
    targets = [p.parts[-3] for p in images]
    
    df = pd.DataFrame({
        'image_path': images,
        'patient': patients,
        'target': targets
    })
    
    print("\nDataset distribution:")
    print(df.groupby('target').size())
    
    # Split by patients to avoid data leakage
    unique_patients = df['patient'].unique()
    
    # First split: train vs (val + test)
    train_patients, testval_patients = train_test_split(
        unique_patients, 
        train_size=train_ratio,
        random_state=seed
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_patients, test_patients = train_test_split(
        testval_patients,
        train_size=val_size,
        random_state=seed
    )
    
    # Create split directories
    create_split_directories(output_dir)
    
    # Copy files to appropriate directories
    splits = {
        'train': train_patients,
        'val': val_patients,
        'test': test_patients
    }
    
    for split_name, patient_list in splits.items():
        split_df = df[df['patient'].isin(patient_list)]
        print(f"\n{split_name.capitalize()} set: {len(split_df)} images, {len(patient_list)} patients")
        print(split_df.groupby('target').size())
        
        # Copy files
        for _, row in split_df.iterrows():
            src = row['image_path']
            dst_dir = os.path.join(output_dir, split_name, row['target'])
            dst = os.path.join(dst_dir, src.name)
            
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    
    print(f"\nDataset split completed. Output: {output_dir}")


def create_model(input_shape=(224, 224, 3), learning_rate=0.0001, model_name='efficientnet'):
    """
    Create model with transfer learning - supports multiple architectures
    
    Args:
        input_shape: Input image shape (height, width, channels)
        learning_rate: Learning rate for optimizer
        model_name: Model architecture - 'efficientnet', 'densenet121', or 'resnet50'
    
    Returns:
        Compiled Keras model
    """
    print(f"Loading {model_name.upper()} base model...")
    
    # Select base model based on model_name
    if model_name == 'efficientnet':
        base_model = EfficientNetB2(
            weights='noisy-student', 
            include_top=False, 
            input_shape=input_shape
        )
        dense_units = 256
    
    elif model_name == 'densenet121':
        from tensorflow.keras.applications import DenseNet121
        base_model = DenseNet121(
            weights='imagenet', 
            include_top=False, 
            input_shape=input_shape
        )
        dense_units = 256
    
    elif model_name == 'resnet50':
        from tensorflow.keras.applications import ResNet50
        base_model = ResNet50(
            weights='imagenet', 
            include_top=False, 
            input_shape=input_shape
        )
        dense_units = 512  # ResNet typically needs more units
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: efficientnet, densenet121, resnet50")
    
    # Build model using Sequential API
    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(dense_units, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dense(3, activation='softmax')
    ])
    
    # Make all layers trainable
    for layer in model.layers:
        layer.trainable = True
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    
    print(f"Model created successfully with {model.count_params():,} parameters")
    
    return model


def create_data_generators(train_dir, val_dir, image_size=(224, 224), batch_size=16):
    """
    Create data generators with augmentation for training and validation
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical'
    )
    
    return train_generator, validation_generator


def create_callbacks(model_file_path, early_stop_patience=20, lr_patience=10):
    """
    Create training callbacks
    """
    # Early stopping
    early_stop = EarlyStopping(
        patience=early_stop_patience,
        verbose=1,
        mode='auto',
        restore_best_weights=True
    )
    
    # Learning rate reduction
    learn_control = ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=lr_patience,
        verbose=1,
        factor=0.2,
        min_lr=0.000001
    )
    
    # Model checkpoint
    checkpoints = ModelCheckpoint(
        model_file_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    return [early_stop, learn_control, checkpoints]


def plot_training_history(history, output_dir='./'):
    """
    Plot training and validation accuracy/loss
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150)
    print(f"Training history plot saved to {os.path.join(output_dir, 'training_history.png')}")
    plt.close()


def evaluate_model(model, generator, class_names, output_dir='./', dataset_name='validation'):
    """
    Evaluate model and generate confusion matrix and classification report
    """
    generator.reset()
    steps = int(np.ceil(generator.n / generator.batch_size))
    
    # Predictions
    pred = model.predict(generator, steps=steps, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)
    
    # Get true labels
    true_classes = generator.classes
    
    # Truncate predictions to match true labels (in case of rounding)
    if len(predicted_class_indices) > len(true_classes):
        predicted_class_indices = predicted_class_indices[:len(true_classes)]
    
    # Calculate metrics
    accuracy = np.mean(predicted_class_indices == true_classes)
    print(f"\n{dataset_name} Accuracy: {accuracy*100:.2f}%")
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_class_indices)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{dataset_name}.png'), dpi=150)
    print(f"Confusion matrix saved to {os.path.join(output_dir, f'confusion_matrix_{dataset_name}.png')}")
    plt.close()
    
    # Classification report
    cr = classification_report(true_classes, predicted_class_indices, target_names=class_names, digits=4)
    print(f"\n--- Classification Report ({dataset_name}) ---\n")
    print(cr)
    
    # Save classification report
    with open(os.path.join(output_dir, f'classification_report_{dataset_name}.txt'), 'w') as f:
        f.write(f"Classification Report - {dataset_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(cr)
    
    return accuracy, cm, cr


def test_model(model_path, test_dir, image_size=(224, 224), batch_size=16, output_dir='./'):
    """
    Test the trained model on test dataset
    """
    print(f"\nLoading model from {model_path}")
    model = load_model(model_path)
    
    # Create test generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    class_names = list(test_generator.class_indices.keys())
    
    # Evaluate
    evaluate_model(model, test_generator, class_names, output_dir, dataset_name='test')


def main(base_dir, output_dir='./output', split_data=True, 
         epochs=60, batch_size=16, learning_rate=0.0001, image_size=224,
         early_stop_patience=20, lr_patience=10, model_name='efficientnet'):
    """
    Main training function
    """
    print("=" * 80)
    print("COVID-19 CT Scan Multiclass Classifier")
    print(f"Using {model_name.upper()} Transfer Learning")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    # Split dataset if needed
    split_dir = os.path.join(output_dir, 'train-test')
    if split_data or not os.path.exists(split_dir):
        split_dataset(base_dir, split_dir)
    
    # Define directories
    train_dir = os.path.join(split_dir, 'train')
    val_dir = os.path.join(split_dir, 'val')
    test_dir = os.path.join(split_dir, 'test')
    
    # Verify directories exist
    for d in [train_dir, val_dir, test_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Directory not found: {d}")
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Model architecture: {model_name}")
    print(f"  Data directory: {base_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Early stop patience: {early_stop_patience}")
    print(f"  LR reduce patience: {lr_patience}")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\nGPU available: {len(gpus)} GPU(s)")
        for gpu in gpus:
            print(f"  {gpu}")
    else:
        print("\nNo GPU available, using CPU")
    
    # Create data generators
    print("\nCreating data generators...")
    image_size_tuple = (image_size, image_size)
    train_generator, validation_generator = create_data_generators(
        train_dir, val_dir, image_size_tuple, batch_size
    )
    
    print(f"Training samples: {train_generator.n}")
    print(f"Validation samples: {validation_generator.n}")
    print(f"Classes: {list(train_generator.class_indices.keys())}")
    
    # Create model
    print(f"\nCreating {model_name.upper()} model...")
    model = create_model(
        input_shape=(image_size, image_size, 3), 
        learning_rate=learning_rate,
        model_name=model_name
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Create callbacks
    model_file_path = os.path.join(output_dir, f'{model_name}_model.h5')
    callbacks = create_callbacks(model_file_path, early_stop_patience, lr_patience)
    
    # Calculate steps (use ceiling to ensure all samples are used)
    steps_per_epoch = int(np.ceil(train_generator.n / batch_size))
    validation_steps = int(np.ceil(validation_generator.n / batch_size))
    
    # Train model
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nTraining completed!")
    
    # Plot training history
    print("\nGenerating training history plots...")
    plot_training_history(history, output_dir)
    
    # Load best model
    print(f"\nLoading best model from {model_file_path}")
    model = load_model(model_file_path)
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("Evaluating on validation set...")
    print("=" * 80)
    class_names = list(validation_generator.class_indices.keys())
    evaluate_model(model, validation_generator, class_names, output_dir, dataset_name='validation')
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("Evaluating on test set...")
    print("=" * 80)
    test_model(model_file_path, test_dir, image_size_tuple, batch_size, output_dir)
    
    print("\n" + "=" * 80)
    print("Training and evaluation completed successfully!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train COVID-19 CT Scan Multiclass Classifier using Transfer Learning'
    )
    
    # Required arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the dataset directory containing Covid/Healthy/Others folders')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for models and results (default: ./output)')
    parser.add_argument('--model', type=str, default='efficientnet',
                        choices=['efficientnet', 'densenet121', 'resnet50'],
                        help='Model architecture to use (default: efficientnet)')
    parser.add_argument('--split_data', action='store_true',
                        help='Split the dataset into train/val/test (default: False, use existing split)')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs (default: 60)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for training (default: 224)')
    parser.add_argument('--early_stop_patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='Learning rate reduction patience (default: 10)')
    
    args = parser.parse_args()
    
    # Run training
    main(
        base_dir=args.data_dir,
        output_dir=args.output_dir,
        split_data=args.split_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        early_stop_patience=args.early_stop_patience,
        lr_patience=args.lr_patience,
        model_name=args.model
    )
