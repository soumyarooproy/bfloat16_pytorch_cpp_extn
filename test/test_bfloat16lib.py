import torch
import torch.utils.cpp_extension

from bfloat16_pytorch_cpp_extn import load_bfloat16lib

import unittest

class TestBFloat16Tensor(unittest.TestCase):

    def test_bfloat16_tensor_empty(self):
        b = torch.empty(2, 2, dtype=torch.bfloat16)
        self.assertEqual(b.size(), torch.Size([2, 2]))
        self.assertEqual(b.type(), 'torch.BFloat16Tensor')

    def test_bfloat16_tensor_convert_from_float(self):
        f = torch.randn(2, 3)
        b = f.to(dtype=torch.bfloat16)
        self.assertEqual(b.size(), torch.Size([2, 3]))
        self.assertEqual(b.type(), 'torch.BFloat16Tensor')
        self.assertEqual(b.dtype, torch.bfloat16)

    def test_bfloat16_tensor_convert_to_float(self):
        b = torch.randn(2, 3).to(dtype=torch.bfloat16)
        f = b.float()
        self.assertEqual(f.size(), torch.Size([2, 3]))
        self.assertEqual(f.type(), 'torch.FloatTensor')
        self.assertEqual(f.dtype, torch.float32)

if __name__ == '__main__':
    load_bfloat16lib()

    unittest.main()
