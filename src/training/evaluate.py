import torch
from src.data.dataloader import get_dataloader
from src.transforms.transforms import get_transforms

def evaluate_model(model, data_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    accuracy = running_corrects.double() / len(data_loader.dataset)
    print(f'Accuracy: {accuracy:.4f}')

def main():
    root_dir = '/home/hamideh/Dataset/Chinese_dataset/gastritis-data/LCI'
    batch_size = 32
    num_workers = 4

    transform = get_transforms()
    _, val_loader = get_dataloader(root_dir, batch_size, num_workers, transform=transform)

    model = SimpleCNN(num_classes=3)
    model.load_state_dict(torch.load('path/to/your/model.pth'))

    evaluate_model(model, val_loader)

if __name__ == "__main__":
    main()
