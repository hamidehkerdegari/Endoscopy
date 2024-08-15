import torchvision.transforms as transforms
from src.data.dataset import EndoscopyDataset

# Define any transforms if needed
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for visualization
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# Create an instance of the dataset
dataset = EndoscopyDataset(root_dir='/home/hamideh/Dataset/Chinese_dataset/gastritis-data/LCI', transform=transform)

# Visualize and save 9 random samples from the dataset
dataset.visualize_samples(num_samples=9, save_path='samples.png')
