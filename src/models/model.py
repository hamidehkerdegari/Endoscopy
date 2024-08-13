#### Simple CNN implementation ####
# import torch.nn as nn
# import torch.nn.functional as F

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=3):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.fc1 = nn.Linear(64 * 28 * 28, 512)
#         self.fc2 = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, 64 * 28 * 28)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


import torch
import torch.nn as nn
from torchvision import models

class VGGClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(VGGClassifier, self).__init__()
        # Load the pretrained VGG16 model
        self.vgg = models.vgg16(pretrained=True)
        # Freeze the convolutional layers
        for param in self.vgg.features.parameters():
            param.requires_grad = False

        # Replace the classifier part with a new one
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)

    def forward(self, x):
        return self.vgg(x)
