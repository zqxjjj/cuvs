import numpy as np
import torch
import my_kernel

kv_head_num = 8
seq_len = 4096
dim = 128

input_keys = torch.randn((kv_head_num, seq_len, dim), dtype=torch.float16, device='cuda')
input_values = torch.randn((kv_head_num, seq_len, dim), dtype=torch.float16, device='cuda')

n_clusters = 16
n_segments = 4

# Print the first 10 elements of input_keys and input_values
print("First 10 elements of input_keys:")
print(input_keys.flatten()[:10])

print("First 10 elements of input_values:")
print(input_values.flatten()[:10])

# Create sum_values to store element-wise addition of first 10 elements
sum_values = input_keys.flatten()[:10] + input_values.flatten()[:10]
print("Sum of first 10 elements (element-wise addition):")
print(sum_values)
print("================================================")
# transfer to CUDA
c = my_kernel.kernel_load(input_keys, input_values, n_clusters, n_segments)
print(c)