import torch
import torch.utils.cpp_extension

import os

def load_bfloat16lib():
    cpp_src = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'bfloat16lib.cpp')
    torch.utils.cpp_extension.load(name='bfloat16lib', sources=cpp_src, verbose=True)
