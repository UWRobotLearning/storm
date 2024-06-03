import torch
import gc

def free_gpu_memory():
    print("Initial GPU memory usage:")
    print_gpu_memory_usage()
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    
    print("GPU memory usage after emptying cache:")
    print_gpu_memory_usage()
    
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                del obj
        except:
            pass
    
    print("GPU memory usage after deleting tensors:")
    print_gpu_memory_usage()

def print_gpu_memory_usage():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB reserved, {torch.cuda.memory_allocated(i) / 1e9:.2f} GB allocated")

if __name__ == "__main__":
    free_gpu_memory()
