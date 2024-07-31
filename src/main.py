# from src.data.dataset import EndoscopyDataset
# from src.data.transforms import get_transforms

# def main():
#     root_dir = '/home/hamideh/Dataset/Chinese_dataset/gastritis-data/LCI'
#     transform = get_transforms()
    
#     dataset = EndoscopyDataset(root_dir=root_dir, transform=transform)
    
#     # Visualize and save some samples
#     dataset.visualize_samples(num_samples=9, save_path='sample_visualization.png')

# if __name__ == "__main__":
#     main()

from src.data.dataset import EndoscopyDataset
from src.data.transforms import get_transforms

def main():
    root_dir = '/home/hamideh/Dataset/Chinese_dataset/gastritis-data/LCI'
    transform = get_transforms()
    
    dataset = EndoscopyDataset(root_dir=root_dir, transform=transform)
    
    # Visualize and save some samples with blurred corners
    dataset.visualize_samples(num_samples=9, save_path='sample_visualization.png')

if __name__ == "__main__":
    main()
