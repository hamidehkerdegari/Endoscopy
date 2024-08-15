# import numpy as np
# import torch.optim as optim
# from src.data.dataloader import get_dataloader
# from src.models.model import VGGClassifier
# from src.models.model import SimpleCNN
# from src.data.transforms import get_transforms
# from src.training.utils import train_model
# from sklearn.utils.class_weight import compute_class_weight
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def plot_metrics(history):
#     epochs = range(1, len(history['train_loss']) + 1)
    
#     plt.figure(figsize=(14, 5))

#     # Plot Loss
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, history['train_loss'], 'b', label='Training Loss')
#     plt.plot(epochs, history['val_loss'], 'r', label='Validation Loss')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()

#     # Plot Accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, history['train_accuracy'], 'b', label='Training Accuracy')
#     plt.plot(epochs, history['val_accuracy'], 'r', label='Validation Accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig("performance.png")


# def main():
#     root_dir = '/home/hamideh/Dataset/Chinese_dataset/gastritis-data/LCI'
#     batch_size = 32
#     num_workers = 4
#     num_epochs = 2
#     learning_rate = 0.0001

#     transform = get_transforms()

#     # Get dataloaders and the train_dataset
#     train_loader, val_loader, train_dataset = get_dataloader(root_dir, batch_size, num_workers, transform=transform)

#     # Compute class weights based on the training dataset
#     train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
#     class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
#     class_weights = torch.tensor(class_weights, dtype=torch.float)

#     # Define the model and loss function with weights
#     model = SimpleCNN()
#     #model = VGGClassifier(pretrained=True)
#     criterion = torch.nn.BCELoss(class_weights)

#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     # Train the model
#     trained_model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

#     # Plot the metrics after training
#     plot_metrics(history)

# if __name__ == "__main__":
#     main()


import numpy as np
import torch.optim as optim
from src.data.dataloader import get_dataloader
from src.models.model import VGGClassifier
from src.models.model import SimpleCNN
from src.data.transforms import get_transforms
from src.training.utils import train_model
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    batch_size = 32
    num_workers = 4
    num_epochs = 2
    learning_rate = 0.0001

    transform = get_transforms()

    # Get dataloaders and the train_dataset
    train_loader, val_loader, train_dataset = get_dataloader(root_dir, batch_size, num_workers, transform=transform)

    # Compute class weights based on the training dataset
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Use class weights to compute pos_weight for BCEWithLogitsLoss
    pos_weight = class_weights[1] / class_weights[0]

    # Define the model and loss function with weights
    model = SimpleCNN()
    #model = VGGClassifier(pretrained=True).to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    trained_model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    # Plot the metrics after training
    plot_metrics(history)

if __name__ == "__main__":
    main()

