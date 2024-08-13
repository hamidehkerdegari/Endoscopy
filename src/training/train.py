import torch
import torch.optim as optim
from src.data.dataloader import get_dataloader
#from src.models.model import SimpleCNN
from src.models.model import VGGClassifier
from src.models.loss import get_loss
from src.data.transforms import get_transforms
from src.training.utils import train_model

def main():
    root_dir = '/home/hamideh/Dataset/Chinese_dataset/gastritis-data/LCI'
    batch_size = 16
    num_workers = 4
    num_epochs = 10
    learning_rate = 0.0001

    transform = get_transforms()
    train_loader, val_loader = get_dataloader(root_dir, batch_size, num_workers, transform=transform)

    model = VGGClassifier(num_classes=3)
    criterion = get_loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

if __name__ == "__main__":
    main()
