import numpy as np
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./data", help='The directory where the data is stored')
    parser.add_argument('--model', type=str, default="llama3_8b", help='The model to use')
    parser.add_argument('--layer_idx', type=int, default="10", help='The layer index to use')
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    model = args.model
    layer_idx = args.layer_idx
    
    # Load key states
    key_file = f'{data_dir}/{model}/key_states_layer_{layer_idx}.npy'
    layer_key = np.load(key_file)  # [bsz, group, seq_len, dim]
    
    print(f"Loaded key shape: {layer_key.shape}")
    
    # Extract and prepare data for testing
    # Reshape to get [kv_head_num, seq_len, dim] format
    # Assuming layer_key has shape [bsz, group, seq_len, dim]
    # We use all heads from the first batch
    keys = layer_key[0]  # Get first batch
    kv_head_num, seq_len, dim = keys.shape
    
    print(f"Preparing data with shape: [{kv_head_num}, {seq_len}, {dim}]")
    
    # Apply centering technique
    for head in range(kv_head_num):
        column_means = np.mean(keys[head], axis=0)
        keys[head] = keys[head] - column_means
    
    # Convert to float16 for compatibility with half precision in CUDA
    if keys.dtype != np.float16:
        keys = keys.astype(np.float16)
    
    # Transfer to GPU
    keys_tensor = torch.from_numpy(keys).to("cuda")
    print(f"Transferred tensor to GPU with shape: {keys_tensor.shape}")
    
    # Import the module after preparing the data
    # This assumes ivf_flat_16 is the name of the compiled extension
    try:
        import ivf_flat_16
        print("Testing IVF-Flat index building...")
        ivf_flat_16.build_test(keys_tensor)
    except ImportError as e:
        print(f"Error importing module: {e}")
        print("Make sure to compile the CUDA extension properly.")

if __name__ == "__main__":
    main()