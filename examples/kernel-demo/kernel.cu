#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

// CUDA kernel for elementwise addition of the first 10 elements
__global__ void elementwise_add_kernel(__half* keys, __half* values, __half* sum, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        sum[idx] = __hadd(keys[idx], values[idx]);
    }
}

torch::Tensor kernel_load(
    torch::Tensor& input_keys,      // [kv_head_num, seq_len, dim], gpu_tensor
    torch::Tensor& input_values,    // [kv_head_num, seq_len, dim], gpu_tensor
    int n_clusters,
    int n_segments
) {
    int kv_head_num = input_keys.size(0);
    int seq_len = input_keys.size(1);
    int dim = input_keys.size(2);

    // get the pointer of input tensor
    __half* key_ptr = reinterpret_cast<__half*>(input_keys.data_ptr<at::Half>());
    __half* value_ptr = reinterpret_cast<__half*>(input_values.data_ptr<at::Half>());

    // for test
    printf("input tensor size: %d x %d x %d\n", kv_head_num, seq_len, dim);

    // Number of elements to add (first 10)
    const int num_elements = 10;
    
    // Create a tensor on GPU to store results
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(input_keys.device());
    torch::Tensor sum_KV = torch::empty({num_elements}, options);
    __half* sum_ptr = reinterpret_cast<__half*>(sum_KV.data_ptr<at::Half>());
    
    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    elementwise_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(key_ptr, value_ptr, sum_ptr, num_elements);
    
    // Synchronize to ensure computation is done
    cudaDeviceSynchronize();
    
    // Transfer result to CPU
    torch::Tensor cpu_sum_KV = sum_KV.to(torch::kCPU);
    
    // build index kernel (example)
    // Index* index = new Index(n_clusters, n_segments);
    // index->build_index(key_ptr, value_ptr, kv_head_num, seq_len, dim);

    // get_labels (example)
    // int* labels = index->get_labels();   # [kv_head_num, seq_len], cpu

    // get centroids and value_sum (example)
    // __half* centroids, value_sum;
    // centroids, value_sum = index->get_centroids();   // [kv_head_num, n_clusters, dim], cpu
    // you can use the following code to wrap the centroids and value_sum and return torch.tensor
    // torch::Tensor centroids_tensor = torch::from_blob(centroids, {kv_head_num, n_clusters, dim}, torch::kFloat16);
    // torch::Tensor value_sum_tensor = torch::from_blob(value_sum, {kv_head_num, n_clusters, dim}, torch::kFloat16);
    // return std::make_tuple(centroids_tensor, value_sum_tensor);

    // delete index and clear memory
    // ......
    
    return cpu_sum_KV;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kernel_load", &kernel_load, "My custom kernel (CUDA)");
}