import torch
import unittest
from one_stage import ConvResidualBlock

class TestConvResidualBlock(unittest.TestCase):
    def setUp(self):
        self.block = ConvResidualBlock(3, 64, 64, 64)
        self.input = torch.randn(1, 3, 32, 32)

    def test_forward(self):
        output = self.block(self.input)
        self.assertEqual(output.shape, torch.Size([1, 128, 32, 32]))

if __name__ == '__main__':
    unittest.main()