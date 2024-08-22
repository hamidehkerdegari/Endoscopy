import os
import torch
import random
import hashlib
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image, to_tensor
import matplotlib.pyplot as plt

class EndoscopyDataset(Dataset):
    def __init__(self, root_dir, augment_dir, transform=None, n_folds=2, epoch=0):
        """
        Args:
            root_dir (string): Directory with all the images.
            augment_dir (string): Directory to save or load augmented images.
            transform (callable, optional): Optional transform to be applied on a sample.
            n_folds (int): Number of folds for augmentation. One fold will be original images, 
                           and the remaining n-1 will be augmented images.
            epoch (int): Current epoch number to determine whether to generate or load images.
        """
        self.root_dir = root_dir
        self.augment_dir = augment_dir
        self.transform = transform
        self.n_folds = n_folds
        self.epoch = epoch
        self.classes = ['IM', 'GA', 'Normal']
        self.views = ['Antrum', 'Angle', 'Cardia', 'Fundus', 'Body']
        self.images = []
        self.labels = []

        # Load and store all images and their labels
        for label in self.classes:
            for view in self.views:
                class_dir = os.path.join(self.root_dir, label, view)
                if not os.path.isdir(class_dir):
                    print(f"Directory {class_dir} does not exist, skipping.")
                    continue
                for file_name in os.listdir(class_dir):
                    if file_name.lower().endswith('.tif'):
                        image_path = os.path.join(class_dir, file_name)
                        self.images.append(image_path)
                        binary_label = 1 if label == 'IM' else 0
                        self.labels.append(binary_label)

        if not self.images:
            raise FileNotFoundError("No .TIF images found in the dataset directories.")

    def _get_augment_filename(self, image_path, fold_idx):
        """
        Generate a unique filename for each augmented image based on the original image path
        and the fold index.
        """
        # Hash the image path and fold index to create a unique filename
        base_name = os.path.basename(image_path)
        hash_val = hashlib.md5(f"{base_name}_{fold_idx}".encode()).hexdigest()
        return os.path.join(self.augment_dir, f"{hash_val}.pt")

    def _save_image(self, image, save_path):
        """ Save the Tensor image to the specified path as a torch tensor file. """
        if isinstance(image, torch.Tensor):
            torch.save(image, save_path)
        else:
            raise TypeError("Expected a torch.Tensor object for saving.")

    def _load_image(self, load_path):
        """ Load the torch tensor image from the specified path. """
        return torch.load(load_path)

    def __len__(self):
        return len(self.images) * self.n_folds

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
        # Determine the fold and the original index
        original_idx = idx // self.n_folds
        fold_idx = idx % self.n_folds

        image_path = self.images[original_idx]
        label = self.labels[original_idx]
        aug_image_path = self._get_augment_filename(image_path, fold_idx)
        
        if self.epoch == 0 and not os.path.exists(aug_image_path):
            # First epoch, generate and save image
            image = Image.open(image_path)
            image = self.mask_circle_and_strip(image)
        
            if fold_idx == 0:
                # Original image (just resized)
                if self.transform:  #TODO; Chech this logic
                    image = self.transform.transforms[0](image)  # Apply resize only
                    image = self.transform.transforms[3](image)  # Apply Normalize only
                    image = self.transform.transforms[4](image)  # Apply Normalize only
            else:
                # Augmented image
                if self.transform:
                    random.seed(fold_idx)
                    torch.manual_seed(fold_idx)
                    image = self.transform(image)
            # Ensure the loaded image is a tensor
            if isinstance(image, Image.Image):
                image = to_tensor(image)
            self._save_image(image, aug_image_path)
        else:
            # Load the augmented image from disk
            image = self._load_image(aug_image_path)

            # Ensure the loaded image is a tensor
            if isinstance(image, Image.Image):
                image = to_tensor(image)
                

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