import torch
from src.data.dataloader import get_dataloader
from src.transforms.transforms import get_transforms
from src.models.model import VGGClassifier  # Ensure you import the correct model class

def evaluate_model(model, data_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()  # For binary classification, output is a single value per input
            preds = (outputs > 0.5).float()  # Convert probabilities to binary predictions with a threshold of 0.5
            running_corrects += torch.sum(preds == labels.data)

    accuracy = running_corrects.double() / len(data_loader.dataset)
    print(f'Accuracy: {accuracy:.4f}')

def main():
    root_dir = '/home/hamideh/Dataset/Chinese_dataset/gastritis-data/LCI'
    batch_size = 32
    num_workers = 4

    transform = get_transforms()
    _, val_loader = get_dataloader(root_dir, batch_size, num_workers, transform=transform)

    # Initialize the binary classification model
    model = VGGClassifier(pretrained=False)
    model.load_state_dict(torch.load('path/to/your/binary_classification_model.pth'))

    # Evaluate the model
    evaluate_model(model, val_loader)

if __name__ == "__main__":
    main()
