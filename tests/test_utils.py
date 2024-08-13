import unittest
from src.training.utils import train_model

class TestUtils(unittest.TestCase):
    def test_train_model(self):
        self.assertTrue(callable(train_model))

if __name__ == '__main__':
    unittest.main()
