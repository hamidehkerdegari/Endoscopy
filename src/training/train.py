import numpy as np
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataloader import get_dataloader
from src.models.model import VGGClassifier, SimpleCNN
from src.data.transforms import get_transforms
from src.training.utils import train_model

def plot_metrics(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_accuracy'], 'b', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("performance.png")


def main():
    root_dir = '/home/hamideh/Dataset/Chinese_dataset/gastritis-data/LCI'
    augment_dir = 'data/Augmented'  # Directory to save augmented images
    batch_size = 64
    num_workers = 4
    num_epochs = 50
    learning_rate = 0.0001
    n_folds = 1  # Number of folds, including one original and four augmented

    # Define the transformation pipeline
    transform = get_transforms()

    # Train the model over multiple epochs
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}/{num_epochs}...")

        # Get dataloaders with the current epoch
        train_loader, val_loader, train_dataset = get_dataloader(
            root_dir=root_dir,
            augment_dir=augment_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            transform=transform,
            n_folds=n_folds,
            epoch=epoch
        )

        # Compute class weights based on the training dataset
        train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float)

        # Use class weights to compute pos_weight for BCEWithLogitsLoss
        pos_weight = class_weights[1] / class_weights[0]

        # Define the model and loss function with weights
        # model = VGGClassifier(pretrained=True)
        model = SimpleCNN()
        
        model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        trained_model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1)

        # Save the model checkpoint after each epoch
        torch.save(trained_model.state_dict(), f'trained_vgg19_model_epoch_{epoch+1}.pth')

        # Plot the metrics after training
        if epoch == num_epochs - 1:  # Plot only after the last epoch
            plot_metrics(history)

if __name__ == "__main__":
    main()
