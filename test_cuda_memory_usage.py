import argparse
import torch
import torch.nn as nn


def test_cuda_memory_usage(tensor_size=(8192, 8192)):
    """Test that CUDA memory usage during a simple operation is below a threshold.
    
    Args:
        tensor_size: Size of the test tensor to create
        memory_threshold_mb: Maximum allowed memory usage in MB
    """
    # Skip test if CUDA is not available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    # Clear cache and reset stats before test
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Create a simple ReLU activation
    activation = nn.ReLU().cuda()
    
    # Generate random input
    x = torch.randn(*tensor_size, device="cuda", dtype=torch.float16)
    
    # Perform forward pass through ReLU
    output = activation(x)
    
    # Check peak memory usage
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"Peak memory usage: {peak_memory_mb:.2f} MB")
    
    # Assert memory usage is below threshold
    memory_threshold = tensor_size[0] * tensor_size[1] * 2 * 2 / 1024 / 1024  # 2 bytes per float16 element, 2x for input and output
    assert peak_memory_mb <= memory_threshold + 10, \
        f"Memory usage ({peak_memory_mb:.2f} MB) exceeds threshold ({memory_threshold} MB)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-size", type=int, nargs=2, default=[8192, 8192], 
                        help="Size of the tensor to use (rows, cols)")
    args = parser.parse_args()
    
    test_cuda_memory_usage(
        tensor_size=tuple(args.tensor_size),
    ) 