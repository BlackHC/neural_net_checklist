#%% Matmul reproduction
import torch
from tqdm.auto import tqdm

# # Set PyTorch to use deterministic algorithms
# torch.use_deterministic_algorithms(True)

# # Set the random seed for reproducibility
# torch.manual_seed(42)

# # If using CUDA, set the CUDA random seed as well
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)

# # For MPS (Metal Performance Shaders) on macOS, set the MPS random seed
# if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#     torch.mps.manual_seed(42)

# # Disable CUDA benchmarking for deterministic behavior
# torch.backends.cudnn.benchmark = False

# # Enable CUDA deterministic mode
# torch.backends.cudnn.deterministic = False

#%%
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        original_shape = x.shape
        return self.linear(x.reshape(-1, original_shape[-1])).reshape(original_shape)
    

device = "cpu"

for _ in tqdm(range(100000)):
    model = Model().to(device)
    x = torch.randn(1, 100, 4).to(device)

    results = [model(x[:, :i]) for i in range(1, x.shape[1]+1)]

    for i, result in enumerate(results):
        if i>0:
            previous = results[i-1]
            current = results[i][:, :i]
            assert torch.allclose(previous, current, atol=1e-6), f"Difference at {i}: {previous - current}"
# %%
