import torch
import torch.optim
import torch.utils.data
import typing
import numpy as np
import functools

from tqdm.auto import tqdm

import base64
import io
from PIL import Image


# A supervised batch input is either a (batched) tensor or a function that takes a batch size and returns a tensor of that batch size.
SupervisedBatchProvider = (
    tuple[torch.Tensor, torch.Tensor]
    | typing.Callable[[int | None], tuple[torch.Tensor, torch.Tensor]]
)


# A model factory is a function that returns a new instance of the model.
ModelFactory = typing.Callable[[], torch.nn.Module]


def get_supervised_data(
    supervised_batch: SupervisedBatchProvider, batch_size: int | None = None
):
    """
    Helper function to get a batch of supervised data from a variety of input types.
    
    Args:
        supervised_batch (SupervisedBatchProvider): The training data.
        batch_size (int, optional): The batch size to use.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The inputs and targets.
    """
    if isinstance(supervised_batch, torch.utils.data.DataLoader):
        supervised_batch = tuple(next(iter(supervised_batch)))
    elif isinstance(supervised_batch, torch.utils.data.Dataset):
        # Check if batch_size is provided
        assert batch_size is not None, "batch_size must be provided if supervised_batch is a Dataset"
        
        # Ensure we don't try to access more items than available in the dataset
        assert batch_size <= len(supervised_batch), f"Dummy batch only has {len(supervised_batch)} samples, but batch size is {batch_size}"
        
        # Convert each item to tensor and stack them
        supervised_batch = tuple(torch.stack([torch.as_tensor(supervised_batch[i][j]) for i in range(batch_size)]) for j in range(2))
    
    if isinstance(supervised_batch, tuple):
        if batch_size is None:
            batch_size = supervised_batch[0].shape[0]

        assert (
            supervised_batch[0].shape[0] >= batch_size
        ), f"Dummy batch only has {supervised_batch[0].shape[0]} samples, but batch size is {batch_size}"
        return supervised_batch[0][:batch_size], supervised_batch[1][:batch_size]
    else:
        return supervised_batch(batch_size)


def train_batch(
    model: torch.nn.Module,
    supervised_batch: SupervisedBatchProvider,
    optimizer: torch.optim.Optimizer | None = None,
    loss_fn: torch.nn.Module = None,
    num_steps: int = 512,
    batch_size: int | None = None,
) -> typing.Generator[tuple[float, torch.optim.Optimizer], None, None]:
    """
    Helper function to train a model on a (single)batch of data.
    
    Args:
        model (torch.nn.Module): The model to train.
        supervised_batch (SupervisedBatchProvider): The training data.
        optimizer (torch.optim.Optimizer, optional): The optimizer to use.
        loss_fn (torch.nn.Module, optional): The loss function to use.
        num_steps (int, optional): The number of steps to train for.
        batch_size (int, optional): The batch size to use.

    Returns:
        tuple[float, torch.optim.Optimizer]: The loss and the optimizer.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    if loss_fn is None:
        loss_fn = torch.nn.functional.cross_entropy

    inputs, targets = get_supervised_data(supervised_batch, batch_size)

    model.train()
    for _ in tqdm(range(num_steps)):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        yield (loss.item(), optimizer)



def assert_balanced_classification_cross_entropy_loss_at_init(
    model: torch.nn.Module, supervised_batch: SupervisedBatchProvider, num_classes: int,
):
    """
    Asserts that the loss at initialization is close to the expected loss for a balanced classification problem.

    This function checks if the initial loss of the model is approximately equal to -log(1/num_classes),
    which is the expected loss when the model's predictions are uniformly distributed across all classes.
    A small tolerance is allowed to account for minor variations due to random initialization.
    
    Args:
        model (torch.nn.Module): The model to check.
        supervised_batch (SupervisedBatchProvider): The training data.
        num_classes (int): The number of classes in the dataset.

    Raises:
        AssertionError: If the model's loss at initialization is not close to the expected loss for a balanced classification problem.
    """
    supervised_inputs, supervised_targets = get_supervised_data(supervised_batch, num_classes)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(supervised_inputs)
        loss = torch.nn.functional.cross_entropy(outputs, supervised_targets)

    expected_loss = np.log(num_classes)
    tolerance = 0.75

    print(
        f"Loss at initialization ({loss.item():.4f}) is within expected range ({expected_loss - tolerance:.4f}, {expected_loss + tolerance:.4f})"
    )
    assert (
        loss.item()/expected_loss - 1.0 <= tolerance
    ), f"Loss at initialization ({loss.item():.4f}) is not within expected range ({expected_loss - tolerance:.4f}, {expected_loss + tolerance:.4f})"


def assert_balanced_classification_crossentropy_init_calibrated(
    model: torch.nn.Module,
    supervised_batch: SupervisedBatchProvider,
    num_classes: int,
):
    """
    Asserts that the model is well-calibrated at initialization, i.e. the model's predictions are uniformly distributed across all classes.
    
    This function checks if the model's predictions are close to the expected probability of 1/num_classes for each class.
    
    Args:
        model (torch.nn.Module): The model to check.
        supervised_batch (SupervisedBatchProvider): The training data.
        num_classes (int): The number of classes in the dataset.

    Raises:
        AssertionError: If the model's predictions are not close to the expected probability of 1/num_classes for each class.
    """
    supervised_inputs, _ = get_supervised_data(
        supervised_batch
    )  # Larger batch for better statistics
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        outputs = model(supervised_inputs)
        probs = torch.softmax(outputs, dim=-1)

        # Calculate the expected probability (1/num_classes for balanced dataset)
        expected_prob = 1.0 / num_classes
        # Check if all class probabilities are close to the expected probability
        tolerance = 0.75
        mean_abs_deviation = torch.abs(probs - expected_prob).mean()
        is_balanced = torch.all(mean_abs_deviation / expected_prob <= tolerance)

        if is_balanced:
            print(
                f"Model is well-calibrated at initialization. Mean deviation: {mean_abs_deviation}"
            )
        else:
            print(
                f"Model is not well-calibrated at initialization. Mean deviation: {mean_abs_deviation}"
            )
            print(f"Expected probability: {expected_prob}")

        assert is_balanced, f"Model is not well-calibrated at initialization. Mean deviation: {mean_abs_deviation}"


def assert_forward_batch_independence(
    model: torch.nn.Module, supervised_batch: SupervisedBatchProvider, train_mode: bool = False
):
    """
    Asserts that the model's forward pass is invariant to the order of the inputs in the batch.
    
    This function checks if the model's forward pass is independent of the order of the inputs in the batch.
    
    Args:
        model (torch.nn.Module): The model to check.
        supervised_batch (SupervisedBatchProvider): The training data.
        train_mode (bool, optional): Whether to use the model in training mode.

    Raises:
        AssertionError: If the model's forward pass is not independent of the order of the inputs in the batch.
    """
    # Check that if we take 3 samples  [A, B, C] then:
    # the results for [A], [B], [C] == [A, B, C]
    # the results for [A, B] == [A, B]
    # the results for [B, C] == [B, C]
    # etc.
    supervised_inputs, _ = get_supervised_data(supervised_batch, 3)
    if train_mode:
        model.train()
    else:
        model.eval()

    a, b, c = supervised_inputs.unbind(0)
    ab_out = model(torch.cat([a[None, :], b[None, :]], dim=0))
    bc_out = model(torch.cat([b[None, :], c[None, :]], dim=0))
    ac_out = model(torch.cat([a[None, :], c[None, :]], dim=0))
    abc_out = model(torch.cat([a[None, :], b[None, :], c[None, :]], dim=0))

    with torch.no_grad():
        a_out = ab_out[0]
        b_out = ab_out[1]
        c_out = bc_out[1]

    assert torch.allclose(b_out, bc_out[0])

    assert torch.allclose(a_out, ac_out[0])
    assert torch.allclose(c_out, ac_out[1])

    assert torch.allclose(a_out, abc_out[0])
    assert torch.allclose(b_out, abc_out[1])
    assert torch.allclose(c_out, abc_out[2])


def assert_non_zero_gradients(
    model: torch.nn.Module, supervised_batch: SupervisedBatchProvider
):
    """
    Asserts that the model's gradients are all non-zero.
    
    This function checks if the model's gradients are non-zero for all parameters,
    ensuring that the model is not trivially zeroed out during backpropagation.
    
    Args:
        model (torch.nn.Module): The model to check.
        supervised_batch (SupervisedBatchProvider): The training data.

    Raises:
        AssertionError: If the model's gradients are all zero.
    """
    supervised_inputs, _ = get_supervised_data(supervised_batch, 1)
    model.train()
    model.zero_grad()

    outputs = model(supervised_inputs)
    outputs.sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert torch.any(param.grad != 0), f"Gradient for {name} is all zeros"

    print("All gradients are non-zero.")


def assert_backward_batch_independence(
    model: torch.nn.Module, supervised_batch: SupervisedBatchProvider
):
    """
    Asserts that the model's backward pass is independent of the batch structure.
    
    This function checks if gradients computed for one sample are not affected by other samples in the batch,
    ensuring that the model treats each input independently during backpropagation.
    
    Args:
        model (torch.nn.Module): The model to check.
        supervised_batch (SupervisedBatchProvider): The training data.

    Raises:
        AssertionError: If the model's backward pass is not independent of the batch structure.
    """
    supervised_inputs, _ = get_supervised_data(supervised_batch, 2)
    supervised_inputs = supervised_inputs.clone().detach().requires_grad_(True)

    model.train()
    model.zero_grad()

    # Forward pass
    outputs = model(supervised_inputs)
    
    print(f"Outputs: {outputs.shape}")

    first_output = outputs[0].sum()
    second_output = outputs[1].sum()
    
    # Calculate gradients for the first output with respect to inputs
    first_output.backward(retain_graph=True)

    # Store gradients for the first output
    first_gradients = supervised_inputs.grad.clone()

    # Zero out the gradients
    supervised_inputs.grad.zero_()

    # Calculate gradients for the second output with respect to inputs

    second_output.backward()

    # Store gradients for the second output
    second_gradients = supervised_inputs.grad.clone()

    # Verify that the other gradient is zero each time
    assert torch.allclose(
        first_gradients[1], torch.zeros_like(first_gradients[1])
    ), f"Gradient of first output with respect to second input is not zero: {first_gradients[1]}"
    assert torch.allclose(
        second_gradients[0], torch.zeros_like(second_gradients[0])
    ), f"Gradient of second output with respect to first input is not zero: {second_gradients[0]}"
    # Also verify that the gradient is not zero
    assert torch.any(
        first_gradients != 0
    ), f"Gradient of first output with respect to second input is all zeros: {first_gradients}"
    assert torch.any(
        second_gradients != 0
    ), f"Gradient of second output with respect to first input is all zeros: {second_gradients}"

    print("Backward pass is batch independent.")


def assert_input_independence_baseline_worse(
    model_factory: ModelFactory,
    supervised_batch: SupervisedBatchProvider,
    train_batch_fn: train_batch = None,
    batch_size: int | None = None,
    loss_fn: torch.nn.Module = None,
    relative_loss_threshold: float = 0.1,
    **kwargs,
):
    """
    Asserts that a model trained on real data performs better than a model trained on fake data.
    
    This function creates two identical models: one trained on the actual input data,
    and another trained on fake data (zeros) but with the same targets. It then compares
    their performance on the original data to ensure that the model actually learns from
    the input features rather than just memorizing the targets.

    Args:
        model_factory (ModelFactory): A function that returns a new instance of the model.
        supervised_batch (SupervisedBatchProvider): The training data.
        train_batch_fn (Callable, optional): A function to train the model for one batch.
        batch_size (int, optional): The batch size to use.
        loss_fn (torch.nn.Module, optional): The loss function to use.
        **kwargs: Additional keyword arguments to pass to train_batch_fn.

    Raises:
        AssertionError: If the model trained on real data doesn't outperform the one trained on fake data.
    """
    if train_batch_fn is None:
        train_batch_fn = functools.partial(train_batch, **kwargs)
    else:
        assert kwargs == {}, "train_batch_fn and kwargs are mutually exclusive"
    if loss_fn is None:
        loss_fn = torch.nn.functional.cross_entropy

    regular_model = model_factory()
    supervised_batch = get_supervised_data(supervised_batch, batch_size)

    input_independent_model = model_factory()
    fake_batch = (torch.zeros_like(supervised_batch[0]), supervised_batch[1])
    for regular_loss_opt, _ in zip(
        train_batch_fn(regular_model, supervised_batch, loss_fn=loss_fn),
        train_batch_fn(input_independent_model, fake_batch, loss_fn=loss_fn),
    ):
        regular_loss, _ = regular_loss_opt
        input_independent_loss = loss_fn(
            input_independent_model(supervised_batch[0]), supervised_batch[1]
        )
        if (input_independent_loss - regular_loss) / max(regular_loss, input_independent_loss) > relative_loss_threshold:
            break
        
    print(f"Regular loss: {regular_loss}")
    print(f"Input independent loss: {input_independent_loss}")
    assert (
        regular_loss < input_independent_loss
    ), f"Regular loss ({regular_loss}) is not less than input independent loss ({input_independent_loss})"


def assert_overfit_to_batch(
    model: torch.nn.Module,
    supervised_batch: SupervisedBatchProvider,
    train_batch_fn: train_batch = None,
    batch_size: int | None = None,
    threshold: float = 1e-6,
    max_steps: int = 2**16,
    optimizer: torch.optim.Optimizer | None = None,
    loss_fn: torch.nn.Module = None,
    **kwargs,
):
    """
    Asserts that a model can overfit to a batch of data.
    
    This function trains a model on a batch of data and asserts that the loss decreases
    to a very small value (close to zero) within a reasonable number of steps.
    
    Args:
        model (torch.nn.Module): The model to train.
        supervised_batch (SupervisedBatchProvider): The training data.
        train_batch_fn (Callable, optional): A function to train the model for one batch.
        batch_size (int, optional): The batch size to use.
        threshold (float, optional): The threshold for the loss to reach.
        max_steps (int, optional): The maximum number of steps to train for.
        optimizer (torch.optim.Optimizer, optional): The optimizer to use.
        loss_fn (torch.nn.Module, optional): The loss function to use.
        **kwargs: Additional keyword arguments to pass to train_batch_fn.

    Raises:
        AssertionError: If the model fails to overfit to the batch.
    """
    print("Overfitting to batch...")
    if train_batch_fn is None:
        train_batch_fn = functools.partial(train_batch, **kwargs)
    else:
        assert kwargs == {}, "train_batch_fn and kwargs are mutually exclusive"
    if loss_fn is None:
        loss_fn = torch.nn.functional.cross_entropy

    supervised_inputs, supervised_targets = get_supervised_data(supervised_batch, batch_size)
    supervised_batch = (supervised_inputs, supervised_targets)
    
    optimizer = None
    loss = float("inf")
    for step, (loss, optimizer) in enumerate(train_batch_fn(model, supervised_batch, optimizer, num_steps=max_steps)):
        if loss < threshold:
            print(f"Loss below threshold at step {step}")
            return loss, optimizer  
    assert False, f"Failed to overfit to batch after {max_steps} steps. Final loss: {loss}."


def is_image_tensor(tensor):
    """
    Helper function to check if a tensor is an image tensor.
    
    Args:
        tensor (torch.Tensor): The tensor to check.

    Returns:
        bool: True if the tensor is an image tensor, False otherwise.
    """
    return len(tensor.shape) in [3, 4] and tensor.shape[len(tensor.shape) - 3] in [1, 3, 4]
    
    
class ModelInputs(Exception):
    """Custom exception for model inputs with IPython display support."""
    model_inputs: typing.Any
    
    def __init__(self, model_inputs: typing.Any):
        super().__init__()
        self.model_inputs = model_inputs
    
    def __str__(self):
        return f"{super().__str__()}\nModel inputs: {self.model_inputs}"
    
    def _repr_html_tensor(self, tensor: torch.Tensor):
        """
        Helper method to generate HTML representation of a tensor.
        """
    
        html = f"<p>Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}</p>"
        
        if is_image_tensor(tensor):
            # Ensure the tensor is on CPU and detached
            tensor = tensor.cpu().detach()
            # Convert to batch if necessary
            if len(tensor.shape) == 3:
                tensor = tensor[None]

            # Normalize if necessary
            if tensor.dtype in [torch.uint8, torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                if tensor.dtype == torch.uint8:
                    tensor = tensor.float() / 255.0
                    html += "<p>Note: Image data was not normalized for display.</p>"
                elif tensor.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
                    tensor = tensor.float()
                    if tensor.max() > 1.0 or tensor.min() < 0.0:
                        min_val = tensor.min()
                        max_val = tensor.max()
                        tensor = (tensor - min_val) / (max_val - min_val)
                        html += f"<p>Note: Image data was normalized for display. Min: {min_val}, Max: {max_val}</p>"
                    else:
                        html += "<p>Note: Image data was not normalized for display.</p>"
        
                # Convert to PIL Images
                for image in tensor:
                    if image.shape[0] in [1, 3, 4]:
                        image = image.permute(1, 2, 0)
                    if image.shape[-1] == 1:
                        image = image.squeeze(-1)
                
                    img = Image.fromarray((image.numpy() * 255).astype('uint8'))
                
                    # Convert PIL image to base64
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    html += f'<img src="data:image/png;base64,{img_str}" />'
            else:
                html += f"<p>Note: Dtype {tensor.dtype} not supported.</p>"
                return html
        else:
            # For non-image tensors, display as text
            html += f"<pre>{tensor}</pre>"
        
        return html
    
    def _repr_html_(self):
        """
        IPython display support for HTML representation.
        """
        html = f"<h3>{self.__class__.__name__}: {super().__str__()}</h3>"

        if isinstance(self.model_inputs, torch.Tensor):
            html += self._repr_html_tensor(self.model_inputs)
        elif isinstance(self.model_inputs, (list, tuple)):
            html += "<p><b>List/Tuple Input:</b></p>"
            for i, item in enumerate(self.model_inputs):
                html += f"<p><b>Item {i}:</b></p>"
                html += self._repr_html_tensor(item)
        elif isinstance(self.model_inputs, dict):
            html += "<p><b>Dict Input:</b></p>"
            for key, value in self.model_inputs.items():
                html += f"<p><b>Key: {key}</b></p>"
                html += self._repr_html_tensor(value)
        else:
            html += "<p><b>Other Input Type:</b></p>"
            html += f"<pre>{self.model_inputs}</pre>"
        
        return html
    

def raise_model_inputs(model_inputs: typing.Any):
    """
    Helper function to throw a custom exception with model inputs for IPython display support.
    """
    raise ModelInputs(model_inputs)


def patch_module_raise_inputs(module: torch.nn.Module):
    """
    Patch a module's forward method to throw a custom exception with model inputs for IPython display support.
    """
    module.forward = functools.wraps(module.forward)(raise_model_inputs)
    
      
def assert_all_for_classification_cross_entropy_loss(model_factory: ModelFactory, supervised_batch: SupervisedBatchProvider, num_classes: int):
    """
    Assert all conditions for a classification model with cross-entropy loss.
    
    Args:
        model_factory (ModelFactory): A function that returns a new instance of the model.
        supervised_batch (SupervisedBatchProvider): The training data.
        num_classes (int): The number of classes in the dataset.
    """
    assert_balanced_classification_cross_entropy_loss_at_init(model_factory(), supervised_batch, num_classes)
    assert_balanced_classification_crossentropy_init_calibrated(model_factory(), supervised_batch, num_classes)
    assert_forward_batch_independence(model_factory(), supervised_batch)
    assert_non_zero_gradients(model_factory(), supervised_batch)
    assert_backward_batch_independence(model_factory(), supervised_batch)
    assert_input_independence_baseline_worse(model_factory, supervised_batch)
    assert_overfit_to_batch(model_factory(), supervised_batch)
