#### Simple CNN implementation ####
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(100352, 64)  # Example input size after flattening
        self.fc2 = nn.Linear(64, 1)  # Single output for binary classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



#vgg19 classifier
class VGGClassifier(nn.Module):
    def __init__(self, pretrained=True, dropout_rate=0.5):
        super(VGGClassifier, self).__init__()
        self.vgg19 = models.vgg19(pretrained=pretrained)

        # Modify the classifier to have one dense layer with dropout
        self.vgg19.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),  # Dropout before the dense layer
            nn.Linear(25088, 1)  # Single output neuron for binary classification
        )

    def forward(self, x):
        x = self.vgg19(x)
        return x  # Logits output for BCEWithLogitsLoss



    
# Vision Transformer (ViT) classifier for binary classification
class ViTClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(ViTClassifier, self).__init__()
        # Load the pretrained Vision Transformer model
        self.vit = models.vit_b_16(pretrained=pretrained)

        # Modify the head for binary classification
        # The original head is designed for 1000 classes (ImageNet)
        self.vit.heads = nn.Sequential(
            nn.Linear(self.vit.heads.head.in_features, 1)  # Output 1 feature for binary classification
        )

    def forward(self, x):
        x = self.vit(x)  # Get the raw logits from ViT
        return x  # Return raw logits
    




