import pytest
import torch

import neural_net_checklist.torch_recipe as torch_recipe


def test_get_dummy_supervised_data_tuple():
    supervised_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    result_batch = torch_recipe.get_supervised_data(supervised_batch, 10) 
    assert torch.allclose(result_batch[0], supervised_batch[0][:10])
    assert torch.allclose(result_batch[1], supervised_batch[1][:10])


def test_get_dummy_supervised_data_dataloader():
    dummy_data = torch.randn(100, 10), torch.randint(0, 10, (100,))
    supervised_batch = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(dummy_data[0], dummy_data[1]), batch_size=32)
    result_batch = torch_recipe.get_supervised_data(supervised_batch, 10) 
    assert torch.allclose(result_batch[0], dummy_data[0][:10])
    assert torch.allclose(result_batch[1], dummy_data[1][:10])


def test_get_dummy_supervised_data_dataset():
    dummy_data = torch.randn(100, 10), torch.randint(0, 10, (100,))
    supervised_batch = torch.utils.data.TensorDataset(dummy_data[0], dummy_data[1])
    result_batch = torch_recipe.get_supervised_data(supervised_batch, 10) 
    assert torch.allclose(result_batch[0], dummy_data[0][:10])
    assert torch.allclose(result_batch[1], dummy_data[1][:10])


def test_get_dummy_supervised_data_function():
    dummy_data = torch.randn(32, 10), torch.randint(0, 10, (32,))
    supervised_batch_fn = lambda batch_size: (dummy_data[0][:batch_size], dummy_data[1][:batch_size])
    result_batch = torch_recipe.get_supervised_data(supervised_batch_fn, 10)
    assert torch.allclose(result_batch[0], dummy_data[0][:10])
    assert torch.allclose(result_batch[1], dummy_data[1][:10])


def test_balanced_classification_cross_entropy_loss_at_init_success():
    model = torch.nn.Linear(10, 10)
    supervised_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    torch_recipe.assert_balanced_classification_cross_entropy_loss_at_init(model, supervised_batch, 10)
    
    
def test_balanced_classification_cross_entropy_loss_at_init_failure():
    model = torch.nn.Linear(10, 10)
    model.weight.data = torch.zeros_like(model.weight.data)
    model.bias.data = torch.zeros_like(model.bias.data) 
    model.bias.data[0] = 1000.
    supervised_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    with pytest.raises(AssertionError):
        torch_recipe.assert_balanced_classification_cross_entropy_loss_at_init(model, supervised_batch, 10)
        

def test_balanced_classification_crossentropy_init_calibrated_success():
    model = torch.nn.Linear(10, 10)
    model.bias.data = torch.zeros_like(model.bias.data) 
    supervised_batch = (torch.randn(1000, 10), torch.randint(0, 10, (1000,)))
    torch_recipe.assert_balanced_classification_crossentropy_init_calibrated(model, supervised_batch, 10)
    

def test_balanced_classification_crossentropy_init_calibrated_failure():
    model = torch.nn.Linear(10, 10)
    model.bias.data = torch.zeros_like(model.bias.data) 
    model.bias.data[0] = 1000.
    supervised_batch = (torch.randn(1000, 10), torch.randint(0, 10, (1000,)))
    with pytest.raises(AssertionError):
        torch_recipe.assert_balanced_classification_crossentropy_init_calibrated(model, supervised_batch, 10)
        
        
def test_forward_batch_independence_success():
    model = torch.nn.Linear(10, 10)
    supervised_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    torch_recipe.assert_forward_batch_independence(model, supervised_batch)
    
    
def test_forward_batch_independence_failure():
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(10),
        torch.nn.Linear(10, 10)
    )
    supervised_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    with pytest.raises(AssertionError):
        torch_recipe.assert_forward_batch_independence(model, supervised_batch)
        
    # For completeness: eval mode has no problems here
    torch_recipe.assert_forward_batch_independence(model, supervised_batch, train_mode=False)
    
    
def test_non_zero_gradients_success():
    model = torch.nn.Linear(10, 10)
    supervised_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    torch_recipe.assert_non_zero_gradients(model, supervised_batch)
    
    
def test_non_zero_gradients_failure():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        # Funnn bug.
        torch.nn.Dropout(1.0)
    )
    supervised_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    with pytest.raises(AssertionError):
        torch_recipe.assert_non_zero_gradients(model, supervised_batch)
        
        
def test_backward_batch_independence_success():
    model = torch.nn.Linear(10, 10)
    supervised_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    torch_recipe.assert_backward_batch_independence(model, supervised_batch)
    
    
def test_backward_batch_independence_failure():
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(10),
        torch.nn.Linear(10, 10)
    )
    supervised_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    with pytest.raises(AssertionError):
        torch_recipe.assert_backward_batch_independence(model, supervised_batch)


def test_input_independence_baseline_worse_success():
    def model_factory():
        return torch.nn.Linear(10, 10)
    dummy_input = torch.randn(100, 10)
    dummy_target = (dummy_input.mean(dim=1) > 0).long()
    supervised_batch = (dummy_input, dummy_target)
    torch_recipe.assert_input_independence_baseline_worse(model_factory, supervised_batch)
    
    
def test_input_independence_baseline_worse_failure():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x) * 0
        
    dummy_input = torch.randn(100, 10)
    dummy_target = (dummy_input.mean(dim=1) > 0).long()
    supervised_batch = (dummy_input, dummy_target)
    with pytest.raises(AssertionError):
        torch_recipe.assert_input_independence_baseline_worse(Model, supervised_batch)
        
        
def test_overfit_to_batch_success():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10)
    )
    dummy_input = torch.randn(32, 10)
    dummy_target = torch.randint(0, 10, (32,))
    supervised_batch = (dummy_input, dummy_target)
    
    loss, _ = torch_recipe.assert_overfit_to_batch(
        model, 
        supervised_batch, 
        threshold=1e-3,
        max_steps=10000
    )
    
    assert loss < 1e-3, f"Model failed to overfit. Final loss: {loss}"
    
    # Verify that the model actually learned the batch
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == dummy_target).float().mean()
    
    assert accuracy > 0.99, f"Model did not truly overfit. Accuracy on training batch: {accuracy}"
    
    
def test_patch_module_raise_inputs():
    model = torch.nn.Linear(10, 10)
    torch_recipe.patch_module_raise_inputs(model)
    with pytest.raises(torch_recipe.ModelInputs) as e:
        inputs = torch.randn(10, 10)
        model(inputs)
        
    assert torch.allclose(e.value.model_inputs, inputs)
    