import torch

def conv2d_output_size(size : torch.Tensor, kernel_size, stride) -> int:
    return torch.div((size - (kernel_size - 1) - 1) , stride, rounding_mode='floor') + 1
