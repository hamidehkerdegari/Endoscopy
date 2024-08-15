import os
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class EndoscopyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['IM', 'GA', 'Normal']
        self.views = ['Antrum', 'Angle', 'Cardia', 'Fundus', 'Body']
        self.images = []
        self.labels = []

        for label in self.classes:
            for view in self.views:
                class_dir = os.path.join(self.root_dir, label, view)
                if not os.path.isdir(class_dir):
                    print(f"Directory {class_dir} does not exist, skipping.")
                    continue
                print(f"Checking directory: {class_dir}")
                for file_name in os.listdir(class_dir):
                    print(f"Found file: {file_name}")
                    if file_name.lower().endswith('.tif'):
                        self.images.append(os.path.join(class_dir, file_name))
                        # Convert labels: 1 for 'IM', 0 for 'GA' and 'Normal'
                        binary_label = 1 if label == 'IM' else 0
                        self.labels.append(binary_label)

        if not self.images:
            raise FileNotFoundError("No .TIF images found in the dataset directories.")

    def __len__(self):
        return len(self.images)

    def mask_circle_and_strip(self, image):
        """
        Mask the image to retain only the circular region in the center
        and apply a rectangular mask to cut out the top and bottom strips.
        
        Args:
            image (PIL.Image): The input image to process.
        
        Returns:
            PIL.Image: The processed image with only the circular region retained
                       and top and bottom strips masked out.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL.Image object for masking circle and strips")

        np_image = np.array(image)
        h, w = np_image.shape[:2]
        
        # Define circle parameters
        center = (w // 2, int(h // 2.15))  # Assuming the circle is centered
        radius = int(min(w, h) // 1.63)  # Adjust this radius if needed

        # Create black masks
        mask1 = np.zeros((h, w), dtype=np.uint8)
        mask2 = np.zeros((h, w), dtype=np.uint8)

        # Draw a white filled circle on the first mask
        cv2.circle(mask1, center, radius, 255, -1)

        # Draw a white filled rectangle on the second mask
        top_left = (0, int(h * 0.0))
        bottom_right = (w, int(h * 0.94))
        cv2.rectangle(mask2, top_left, bottom_right, 255, -1)

        # Combine the two masks using bitwise AND
        combined_mask = cv2.bitwise_and(mask1, mask2)

        # Convert mask to 3 channels
        combined_mask = np.stack([combined_mask] * 3, axis=-1)

        # Apply the mask to the image
        masked_image = np_image.copy()
        masked_image[combined_mask == 0] = 0

        # Convert the masked image back to PIL format
        return Image.fromarray(masked_image)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]

        # Mask the image to retain only the circular region and apply rectangular masks
        image = self.mask_circle_and_strip(image)

        if self.transform:
            image = self.transform(image)

        return image, label

    def visualize_samples(self, num_samples=9, save_path='samples.png'):
        """
        Visualize a few samples from the dataset.
        
        Args:
            num_samples (int): Number of samples to visualize.
            save_path (string): Path to save the visualized samples.
        """
        indices = torch.randperm(len(self.images))[:num_samples]
        images = [Image.open(self.images[i]) for i in indices]
        labels = [self.labels[i] for i in indices]

        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            image = images[i]
            image = self.mask_circle_and_strip(image)  # Mask the image for visualization
            if isinstance(image, Image.Image):
                image = np.array(image) / 255.0  # Normalize for visualization

            # Convert label to human-readable form
            label_text = 'IM' if labels[i] == 1 else 'Not IM'
            
            # Display image and label
            ax.imshow(image)
            ax.set_title(f"Label: {label_text}")
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Samples saved to {save_path}")