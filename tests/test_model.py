import unittest
import torch
from src.models.model import SimpleCNN

class TestSimpleCNN(unittest.TestCase):
    def setUp(self):
        self.model = SimpleCNN(num_classes=3)

    def test_forward(self):
        input_tensor = torch.randn(1, 3, 224, 224)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (1, 3))

if __name__ == '__main__':
    unittest.main()
