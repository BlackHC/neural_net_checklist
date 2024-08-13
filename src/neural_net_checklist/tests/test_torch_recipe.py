import pytest
import torch

import neural_net_checklist.torch_recipe as torch_recipe


def test_balanced_classification_cross_entropy_loss_at_init_success():
    model = torch.nn.Linear(10, 10)
    dummy_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    torch_recipe.assert_balanced_classification_cross_entropy_loss_at_init(model, dummy_batch, 10)
    
    
def test_balanced_classification_cross_entropy_loss_at_init_failure():
    model = torch.nn.Linear(10, 10)
    model.weight.data = torch.zeros_like(model.weight.data)
    model.bias.data = torch.zeros_like(model.bias.data) 
    model.bias.data[0] = 1000.
    dummy_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    with pytest.raises(AssertionError):
        torch_recipe.assert_balanced_classification_cross_entropy_loss_at_init(model, dummy_batch, 10)
        

def test_balanced_classification_crossentropy_init_calibrated_success():
    model = torch.nn.Linear(10, 10)
    model.bias.data = torch.zeros_like(model.bias.data) 
    dummy_batch = (torch.randn(1000, 10), torch.randint(0, 10, (1000,)))
    torch_recipe.assert_balanced_classification_crossentropy_init_calibrated(model, dummy_batch, 10)
    

def test_balanced_classification_crossentropy_init_calibrated_failure():
    model = torch.nn.Linear(10, 10)
    model.bias.data = torch.zeros_like(model.bias.data) 
    model.bias.data[0] = 1000.
    dummy_batch = (torch.randn(1000, 10), torch.randint(0, 10, (1000,)))
    with pytest.raises(AssertionError):
        torch_recipe.assert_balanced_classification_crossentropy_init_calibrated(model, dummy_batch, 10)
        
        
def test_forward_batch_independence_success():
    model = torch.nn.Linear(10, 10)
    dummy_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    torch_recipe.assert_forward_batch_independence(model, dummy_batch)
    
    
def test_forward_batch_independence_failure():
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(10),
        torch.nn.Linear(10, 10)
    )
    dummy_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    with pytest.raises(AssertionError):
        torch_recipe.assert_forward_batch_independence(model, dummy_batch)
        
    # For completeness: eval mode has no problems here
    torch_recipe.assert_forward_batch_independence(model, dummy_batch, train_mode=False)
    
    
def test_non_zero_gradients_success():
    model = torch.nn.Linear(10, 10)
    dummy_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    torch_recipe.assert_non_zero_gradients(model, dummy_batch)
    
    
def test_non_zero_gradients_failure():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        # Funnn bug.
        torch.nn.Dropout(1.0)
    )
    dummy_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    with pytest.raises(AssertionError):
        torch_recipe.assert_non_zero_gradients(model, dummy_batch)
        
        
def test_backward_batch_independence_success():
    model = torch.nn.Linear(10, 10)
    dummy_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    torch_recipe.assert_backward_batch_independence(model, dummy_batch)
    
    
def test_backward_batch_independence_failure():
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(10),
        torch.nn.Linear(10, 10)
    )
    dummy_batch = (torch.randn(100, 10), torch.randint(0, 10, (100,)))
    with pytest.raises(AssertionError):
        torch_recipe.assert_backward_batch_independence(model, dummy_batch)


def test_input_independence_baseline_worse_success():
    def model_factory():
        return torch.nn.Linear(10, 10)
    dummy_input = torch.randn(100, 10)
    dummy_target = (dummy_input.mean(dim=1) > 0).long()
    dummy_batch = (dummy_input, dummy_target)
    torch_recipe.assert_input_independence_baseline_worse(model_factory, dummy_batch)
    
    
def test_input_independence_baseline_worse_failure():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x) * 0
        
    dummy_input = torch.randn(100, 10)
    dummy_target = (dummy_input.mean(dim=1) > 0).long()
    dummy_batch = (dummy_input, dummy_target)
    with pytest.raises(AssertionError):
        torch_recipe.assert_input_independence_baseline_worse(Model, dummy_batch)
        
        