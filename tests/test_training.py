import unittest
import torch
from torch.utils.data import DataLoader
from src.models.model import SimpleCNN
from src.training.utils import train_model
from src.models.loss import get_loss

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.randn(100, 3, 224, 224)
        self.labels = torch.randint(0, 3, (100,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class TestTraining(unittest.TestCase):
    def setUp(self):
        self.train_loader = DataLoader(DummyDataset(), batch_size=32, shuffle=True)
        self.val_loader = DataLoader(DummyDataset(), batch_size=32, shuffle=False)
        self.model = SimpleCNN(num_classes=3)
        self.criterion = get_loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def test_training(self):
        trained_model = train_model(self.model, self.train_loader, self.val_loader, self.criterion, self.optimizer, num_epochs=1)
        self.assertIsNotNone(trained_model)

if __name__ == '__main__':
    unittest.main()
