# neural_net_checklist: A Codified Recipe for Training Neural Networks

This repository implements a set of diagnostic assertions based on Andrei Karpathy's blog post ["A Recipe for Training Neural Networks"](https://karpathy.github.io/2019/04/25/recipe/). Instead of a manual checklist, we provide programmatic checks to diagnose and debug neural networks efficiently.

## Why This Exists

Training neural networks can be tricky. This toolkit saves you time by automating common diagnostic steps, allowing you to focus on the interesting parts of your model development.

## Example Output (Causal Transformer)

Batteries included:

```python
from neural_net_checklist import torch_diagnostics

torch_diagnostics.assert_all_for_causal_llm_cross_entropy_loss(
    lambda: CausalTransformer(vocab_size),
    dataloader,
    embedding_layer_name="embedding",
    vocab_size=vocab_size,
    device="cpu",
)
```

Output:

```
🔍 Checking all conditions for a causal LLM model with cross-entropy loss...
🔍 Checking loss at initialization...
Loss at initialization (4.2921) is within 75.0% of expected loss (4.0775)
✅ Loss at initialization is within the expected range.
🔍 Checking calibration at initialization...
Model is well-calibrated at initialization. Mean deviation: 0.00786806270480156
✅ Model is well-calibrated at initialization.
🔍 Checking forward batch independence...
🔧 Replaced 10 norm layers with Identity: {'LayerNorm': {'transformer.layers.3.norm2', 'transformer.layers.2.norm1', 'transformer.layers.0.norm2', 'transformer.layers.4.norm2', 'transformer.layers.0.norm1', 'transformer.layers.2.norm2', 'transformer.layers.1.norm2', 'transformer.layers.4.norm1', 'transformer.layers.1.norm1', 'transformer.layers.3.norm1'}}
✅ Forward batch independence verified.
🔍 Checking forward causal property...
🔧 Replaced 10 norm layers with Identity: {'LayerNorm': {'transformer.layers.3.norm2', 'transformer.layers.2.norm1', 'transformer.layers.0.norm2', 'transformer.layers.4.norm2', 'transformer.layers.0.norm1', 'transformer.layers.2.norm2', 'transformer.layers.1.norm2', 'transformer.layers.4.norm1', 'transformer.layers.1.norm1', 'transformer.layers.3.norm1'}}
✅ Causal independence verified.
🔍 Checking non-zero gradients...
✅ All gradients are non-zero.
🔍 Checking backward batch independence...
🔧 Replaced 10 norm layers with Identity: {'LayerNorm': {'transformer.layers.3.norm2', 'transformer.layers.2.norm1', 'transformer.layers.0.norm2', 'transformer.layers.4.norm2', 'transformer.layers.0.norm1', 'transformer.layers.2.norm2', 'transformer.layers.1.norm2', 'transformer.layers.4.norm1', 'transformer.layers.1.norm1', 'transformer.layers.3.norm1'}}
✅ Backward pass is batch independent.
🔍 Checking backward causal property...
🔧 Replaced 10 norm layers with Identity: {'LayerNorm': {'transformer.layers.3.norm2', 'transformer.layers.2.norm1', 'transformer.layers.0.norm2', 'transformer.layers.4.norm2', 'transformer.layers.0.norm1', 'transformer.layers.2.norm2', 'transformer.layers.1.norm2', 'transformer.layers.4.norm1', 'transformer.layers.1.norm1', 'transformer.layers.3.norm1'}}
✅ Backward pass exhibits causal independence.
🔍 Checking input independence baseline worse...
Loss: 3.125023:   7%|▋         | 9/128 [00:01<00:24,  4.89it/s]
Loss: 2.736129:   7%|▋         | 9/128 [00:02<00:26,  4.48it/s]
Regular loss: 2.736128568649292
Input independent loss: 3.1189355850219727
✅ Input independent baseline is worse.
🔍 Overfitting to the batch...
Loss: 0.099780:   0%|          | 87/65536 [00:07<1:39:13, 10.99it/s]Loss (0.09978045523166656) below threshold (0.1) at step 87.
✅ Model can overfit to batch.
✅ All conditions for a causal LLM model with cross-entropy loss verified.
```

See `examples/causal_transformer.py` for a complete example. (Examples for ResNet on CIFAR10 and LeNet on MNIST are also included.)

## What's Included

We've implemented the following checks:

1. **Verify loss @ init** (for balanced classification tasks)
   ```python
   assert_balanced_classification_cross_entropy_loss_at_init
   ```

2. **Init well** (for balanced classification tasks)
   ```python
   assert_balanced_classification_cross_entropy_loss_at_init
   ```

3. **Non-zero gradients**
   ```python
   assert_non_zero_gradients
   ```

4. **Batch independence**
   - Forward pass (memory-efficient, but note: batchnorm breaks this naturally)
     ```python
     assert_batch_independence_forward
     ```
   - Backward pass (uses more memory, checks gradients)
     ```python
     assert_batch_independence_backward
     ```

5. **Overfit one batch**
   ```python
   assert_overfit_one_batch
   ```

6. **Visualize just before the net**
   ```python
   patch_module_raise_inputs
   ```

## Quick Start

To run all assertions:

- For classification tasks (e.g., computer vision):
  ```python
  assert_all_for_classification_cross_entropy_loss
  ```

- For causal language models:
  ```python
  assert_all_for_llm_cross_entropy_loss
  ```

## Installation

```bash
pip install neural_net_checklist
```

## Usage Example

```python
import neural_net_checklist.torch_diagnostics as torch_diagnostics

# Assume you have a model and a DataLoader
model = YourModel()
train_loader = YourDataLoader()

# Run all checks
torch_diagnostics.assert_all_for_classification_cross_entropy_loss(model, train_loader)
```

See the code and docstrings for more details.

## Contributing

Contributions are welcome! If you have ideas for additional checks or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Happy debugging! Remember, neural nets can be finicky, but with the right tools, we can tame them. 🧠🔧
