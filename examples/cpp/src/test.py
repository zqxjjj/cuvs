import numpy as np
import torch
import argparse
import time
import ivf_flat_perf  # This is the compiled extension from ivf_flat_fp16_example.cu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./data", help='The directory where the data is stored')
    parser.add_argument('--model', type=str, default="llama3_8b", help='The model to use')
    parser.add_argument('--layer_idx', type=int, default="10", help='The layer index to use')
    parser.add_argument('--nlist', type=int, default="4096", help='The number of inverted lists')
    parser.add_argument('--segment_count', type=int, default="64", help='The number of segments for cluster-segment assignment')

    args = parser.parse_args()
    
    data_dir = args.data_dir
    model = args.model
    layer_idx = args.layer_idx
    
    # Load key states
    key_file = f'{data_dir}/{model}/key_states_layer_{layer_idx}.npy'
    layer_key = np.load(key_file)  # [bsz, group, seq_len, dim]
    
    
    # Extract the first group's first head data
    key = layer_key[0][0].astype(np.float32)
    # centering technique from MagicPIG
    column_means = np.mean(key, axis=0)
    data_center_0 = key - column_means
    
    print(f"Number of vectors: {data_center_0.shape[0]}")
    print(f"Vector dimension: {data_center_0.shape[1]}")
    
    # # Only process head[0] (first head in first group) - commented out conflicting code
    # data = layer_key[:, 0, :, :].squeeze()
    # # Apply centering technique from MagicPIG
    # mean_data = np.mean(data, axis=0)
    # data_center_0 = data - mean_data
    
    # # Convert to float32 if not already - commented out conflicting code
    # if data_center_0.dtype != np.float32:
    #     data_center_0 = data_center_0.astype(np.float32)
    
    # Transfer to GPU
    data_tensor = torch.from_numpy(data_center_0).to("cuda")
    
    # Build IVF-Flat index using our CUDA extension
    print(f"Building IVF-Flat index with {args.nlist} clusters and {args.segment_count} segments...")
    ivf_flat_perf.ivf_flat_build_performance_test(
        data_tensor,
        n_lists=args.nlist,
        segment_count=args.segment_count
    )

if __name__ == "__main__":
    main()