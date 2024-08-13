import unittest
from src.data.dataset import EndoscopyDataset

class TestEndoscopyDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = EndoscopyDataset(root_dir='/home/hamideh/Dataset/Chinese_dataset/gastritis-data/LCI')

    def test_length(self):
        self.assertEqual(len(self.dataset), 100)  # Example length

    def test_getitem(self):
        image, label = self.dataset[0]
        self.assertEqual(image.shape, (3, 224, 224))  # Example shape
        self.assertIn(label, [0, 1, 2])  # Example labels

if __name__ == '__main__':
    unittest.main()
