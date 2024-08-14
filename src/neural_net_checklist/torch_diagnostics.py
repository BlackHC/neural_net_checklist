import torch
import torch.optim
import torch.utils.data
import typing
import numpy as np
import functools

from collections import defaultdict
from tqdm.auto import tqdm

import base64
import io
from PIL import Image


DEFAULT_DEVICE = "cpu"
ALL_CLOSE_ATOL = 1e-6


# A supervised batch input is either a (batched) tensor or a function that takes a batch size and returns a tensor of that batch size.
SupervisedBatchProvider = (
    tuple[torch.Tensor, torch.Tensor]
    | typing.Callable[[int | None], tuple[torch.Tensor, torch.Tensor]]
)


# A model factory is a function that returns a new instance of the model.
ModelFactory = typing.Callable[[], torch.nn.Module]


def get_supervised_batch(
    supervised_batch_provider: SupervisedBatchProvider,
    *,
    batch_size: int | None = None,
    device: str = DEFAULT_DEVICE,
):
    """
    Helper function to get a batch of supervised data from a variety of input types.

    Args:
        supervised_batch_provider (SupervisedBatchProvider): The training data.
        batch_size (int, optional): The batch size to use.
        device (str, optional): The device to move the tensors to.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The inputs and targets.
    """
    if isinstance(supervised_batch_provider, torch.utils.data.DataLoader):
        supervised_batch_provider = tuple(next(iter(supervised_batch_provider)))
    elif isinstance(supervised_batch_provider, torch.utils.data.Dataset):
        # Check if batch_size is provided
        assert (
            batch_size is not None
        ), "batch_size must be provided if supervised_batch_provider is a Dataset"

        # Ensure we don't try to access more items than available in the dataset
        assert (
            batch_size <= len(supervised_batch_provider)
        ), f"Dummy batch only has {len(supervised_batch_provider)} samples, but batch size is {batch_size}"

        # Convert each item to tensor and stack them
        supervised_batch_provider = tuple(
            torch.stack(
                [
                    torch.as_tensor(supervised_batch_provider[i][j])
                    for i in range(batch_size)
                ]
            )
            for j in range(2)
        )

    if isinstance(supervised_batch_provider, tuple):
        if batch_size is None:
            batch_size = supervised_batch_provider[0].shape[0]

        assert (
            supervised_batch_provider[0].shape[0] >= batch_size
        ), f"Dummy batch only has {supervised_batch_provider[0].shape[0]} samples, but batch size is {batch_size}"
        supervised_batch = (
            supervised_batch_provider[0][:batch_size],
            supervised_batch_provider[1][:batch_size],
        )
    else:
        supervised_batch = supervised_batch_provider(batch_size)

    return supervised_batch[0].to(device), supervised_batch[1].to(device)


def train_batch(
    model: torch.nn.Module,
    supervised_batch_provider: SupervisedBatchProvider,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    loss_fn: torch.nn.Module = None,
    num_steps: int = 512,
    batch_size: int | None = None,
    device: str = DEFAULT_DEVICE,
) -> typing.Generator[tuple[float, torch.optim.Optimizer], None, None]:
    """
    Helper function to train a model on a (single)batch of data.

    Args:
        model (torch.nn.Module): The model to train.
        supervised_batch_provider (SupervisedBatchProvider): The training data.
        optimizer (torch.optim.Optimizer, optional): The optimizer to use.
        loss_fn (torch.nn.Module, optional): The loss function to use.
        num_steps (int, optional): The number of steps to train for.
        batch_size (int, optional): The batch size to use.
        device (str, optional): The device to move the tensors to.

    Returns:
        tuple[float, torch.optim.Optimizer]: The loss and the optimizer.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    if loss_fn is None:
        loss_fn = torch.nn.functional.cross_entropy

    inputs, targets = get_supervised_batch(
        supervised_batch_provider, batch_size=batch_size, device=device
    )
    model.to(device)

    model.train()
    pbar = tqdm(range(num_steps))
    for _ in pbar:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {loss.item():.6f}")

        yield (loss.item(), optimizer)


def replace_norm_layers_with_identity(model: torch.nn.Module):
    """
    Replaces all norm layers with identity layers.

    Args:
        model (torch.nn.Module): The model to replace the norm layers in.
    """
    replaced_module_infos = defaultdict(set)

    def replace_norm_layer(module, prefix=""):
        for name, child in module.named_children():
            if isinstance(
                child,
                (
                    torch.nn.BatchNorm1d,
                    torch.nn.BatchNorm2d,
                    torch.nn.BatchNorm3d,
                    torch.nn.LayerNorm,
                    torch.nn.GroupNorm,
                ),
            ):
                replaced_module_infos[child.__class__.__name__].add(f"{prefix}.{name}")
                getattr(module, name).forward = lambda x: x
            else:
                replace_norm_layer(child, prefix=f"{prefix}.{name}" if prefix else name)

    replace_norm_layer(model)
    print(
        f"🔧 Replaced {sum(len(v) for v in replaced_module_infos.values())} norm layers with Identity: {dict(replaced_module_infos)}"
    )


def assert_balanced_classification_cross_entropy_loss_at_init(
    model: torch.nn.Module,
    supervised_batch_provider: SupervisedBatchProvider,
    *,
    num_classes: int,
    device: str = DEFAULT_DEVICE,
    rel_tolerance=0.75,
    loss_fn: torch.nn.Module = None,
):
    """
    Asserts that the loss at initialization is close to the expected loss for a balanced classification problem.

    This function checks if the initial loss of the model is approximately equal to -log(1/num_classes),
    which is the expected loss when the model's predictions are uniformly distributed across all classes.
    A small tolerance is allowed to account for minor variations due to random initialization.

    Args:
        model (torch.nn.Module): The model to check.
        supervised_batch_provider (SupervisedBatchProvider): The training data.
        num_classes (int): The number of classes in the dataset.
        device (str, optional): The device to move the tensors to.

    Raises:
        AssertionError: If the model's loss at initialization is not close to the expected loss for a balanced classification problem.
    """
    print("🔍 Checking loss at initialization...")
    
    if loss_fn is None:
        loss_fn = torch.nn.functional.cross_entropy

    supervised_inputs, supervised_targets = get_supervised_batch(
        supervised_batch_provider, batch_size=min(128, num_classes), device=device
    )
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(supervised_inputs)
        loss = loss_fn(outputs, supervised_targets)

    expected_loss = np.log(num_classes)

    print(
        f"Loss at initialization ({loss.item():.4f}) is within {rel_tolerance * 100:.1f}% of expected loss ({expected_loss:.4f})"
    )
    assert (
        loss.item() / expected_loss - 1.0 <= rel_tolerance
    ), f"Loss at initialization ({loss.item():.4f}) is not within {rel_tolerance * 100:.1f}% of expected loss ({expected_loss:.4f})"
    print("✅ Loss at initialization is within the expected range.")
    

def assert_balanced_classification_cross_entropy_init_calibrated(
    model: torch.nn.Module,
    supervised_batch_provider: SupervisedBatchProvider,
    *,
    num_classes: int,
    device: str = DEFAULT_DEVICE,
    rel_tolerance: float = 1.0,
):
    """
    Asserts that the model is well-calibrated at initialization, i.e. the model's predictions are uniformly distributed across all classes.

    This function checks if the model's predictions are close to the expected probability of 1/num_classes for each class.

    Args:
        model (torch.nn.Module): The model to check.
        supervised_batch_provider (SupervisedBatchProvider): The training data.
        num_classes (int): The number of classes in the dataset.
        device (str, optional): The device to move the tensors to.

    Raises:
        AssertionError: If the model's predictions are not close to the expected probability of 1/num_classes for each class.
    """
    print("🔍 Checking calibration at initialization...")
    supervised_inputs, _ = get_supervised_batch(
        supervised_batch_provider, device=device
    )  # Larger batch for better statistics
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        outputs = model(supervised_inputs)
        probs = torch.softmax(outputs, dim=-1)

        # Calculate the expected probability (1/num_classes for balanced dataset)
        expected_prob = 1.0 / num_classes
        # Check if all class probabilities are close to the expected probability

        mean_abs_deviation = torch.abs(probs - expected_prob).mean()
        is_balanced = torch.all(mean_abs_deviation / expected_prob <= rel_tolerance)

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
    print("✅ Model is well-calibrated at initialization.")
    

def assert_forward_batch_independence(
    model_factory: ModelFactory,
    supervised_batch_provider: SupervisedBatchProvider,
    *,
    train_mode: bool = False,
    device: str = DEFAULT_DEVICE,
):
    """
    Asserts that the model's forward pass is invariant to the order of the inputs in the batch.

    This function checks if the model's forward pass is independent of the order of the inputs in the batch.

    Args:
        model (torch.nn.Module): The model to check.
        supervised_batch_provider (SupervisedBatchProvider): The training data.
        train_mode (bool, optional): Whether to use the model in training mode.
        device (str, optional): The device to move the tensors to.

    Raises:
        AssertionError: If the model's forward pass is not independent of the order of the inputs in the batch.
    """
    print("🔍 Checking forward batch independence...")
    supervised_inputs, _ = get_supervised_batch(
        supervised_batch_provider, batch_size=3, device=device
    )
    model = model_factory().to(device)
    replace_norm_layers_with_identity(model)
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

    assert torch.allclose(
        b_out, bc_out[0], atol=ALL_CLOSE_ATOL
    ), f"b_out: {b_out}, bc_out[0]: {bc_out[0]}, diff: {b_out - bc_out[0]}"

    assert torch.allclose(
        a_out, ac_out[0], atol=ALL_CLOSE_ATOL
    ), f"a_out: {a_out}, ac_out[0]: {ac_out[0]}, diff: {a_out - ac_out[0]}"
    assert torch.allclose(
        c_out, ac_out[1], atol=ALL_CLOSE_ATOL
    ), f"c_out: {c_out}, ac_out[1]: {ac_out[1]}, diff: {c_out - ac_out[1]}"

    assert torch.allclose(
        a_out, abc_out[0], atol=ALL_CLOSE_ATOL
    ), f"a_out: {a_out}, abc_out[0]: {abc_out[0]}, diff: {a_out - abc_out[0]}"
    assert torch.allclose(
        b_out, abc_out[1], atol=ALL_CLOSE_ATOL
    ), f"b_out: {b_out}, abc_out[1]: {abc_out[1]}, diff: {b_out - abc_out[1]}"
    assert torch.allclose(
        c_out, abc_out[2], atol=ALL_CLOSE_ATOL
    ), f"c_out: {c_out}, abc_out[2]: {abc_out[2]}, diff: {c_out - abc_out[2]}"
    print("✅ Forward batch independence verified.")
    
    

def assert_forward_causal_property(
    model_factory: ModelFactory,
    supervised_batch_provider: SupervisedBatchProvider,
    *,
    train_mode: bool = False,
    device: str = DEFAULT_DEVICE,
):
    """
    Asserts that the model's forward pass exhibits causal independence.

    This function checks if later tokens only depend on earlier ones, assuming causal_dim = 1.

    Args:
        model_factory (ModelFactory): A function that returns a new instance of the model.
        supervised_batch_provider (SupervisedBatchProvider): The training data.
        train_mode (bool, optional): Whether to use the model in training mode.
        device (str, optional): The device to move the tensors to.

    Raises:
        AssertionError: If the model's forward pass does not exhibit causal independence.
    """
    print("🔍 Checking forward causal property...")
    supervised_inputs, _ = get_supervised_batch(
        supervised_batch_provider, batch_size=1, device=device
    )
    model = model_factory().to(device)
    replace_norm_layers_with_identity(model)
    if train_mode:
        model.train()
    else:
        model.eval()

    # Assume input shape is (batch_size, sequence_length, ...)
    sequence_length = min(supervised_inputs.shape[1], 16)

    # Generate outputs for progressively longer input sequences
    outputs = []
    for i in range(1, sequence_length + 1):
        with torch.no_grad():
            output = model(supervised_inputs[:, :i])
        outputs.append(output)

    # Check causal property
    for i in range(1, sequence_length):
        assert torch.allclose(
            outputs[i - 1][:, :i], outputs[i][:, :i], atol=ALL_CLOSE_ATOL
        ), f"Causal independence violated at position {i}: {outputs[i-1][:, :i]} != {outputs[i][:, :i]}"

    print("✅ Causal independence verified.")


def assert_non_zero_gradients(
    model: torch.nn.Module,
    supervised_batch_provider: SupervisedBatchProvider,
    *,
    device: str = DEFAULT_DEVICE,
):
    """
    Asserts that the model's gradients are all non-zero.

    This function checks if the model's gradients are non-zero for all parameters,
    ensuring that the model is not trivially zeroed out during backpropagation.

    Args:
        model (torch.nn.Module): The model to check.
        supervised_batch_provider (SupervisedBatchProvider): The training data.
        device (str, optional): The device to move the tensors to.

    Raises:
        AssertionError: If the model's gradients are all zero.
    """
    print("🔍 Checking non-zero gradients...")
    
    supervised_inputs, _ = get_supervised_batch(
        supervised_batch_provider, batch_size=1, device=device
    )
    model.to(device)
    model.train()
    model.zero_grad()

    outputs = model(supervised_inputs)
    outputs.sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert torch.any(param.grad != 0), f"Gradient for {name} is all zeros"

    print("✅ All gradients are non-zero.")


def assert_backward_batch_independence(
    model_factory: ModelFactory,
    supervised_batch_provider: SupervisedBatchProvider,
    *,
    device: str = DEFAULT_DEVICE,
):
    """
    Asserts that the model's backward pass is independent of the batch structure.

    This function checks if gradients computed for one sample are not affected by other samples in the batch,
    ensuring that the model treats each input independently during backpropagation.

    Args:
        model (torch.nn.Module): The model to check.
        supervised_batch_provider (SupervisedBatchProvider): The training data.
        device (str, optional): The device to move the tensors to.

    Raises:
        AssertionError: If the model's backward pass is not independent of the batch structure.
    """
    print("🔍 Checking backward batch independence...")
    
    supervised_inputs, _ = get_supervised_batch(
        supervised_batch_provider, batch_size=2, device=device
    )
    supervised_inputs = supervised_inputs.clone().detach().requires_grad_(True)
    model = model_factory().to(device)
    replace_norm_layers_with_identity(model)

    model.train()
    model.zero_grad()

    # Forward pass
    outputs = model(supervised_inputs)

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
        first_gradients[1], torch.zeros_like(first_gradients[1]), atol=ALL_CLOSE_ATOL
    ), f"Gradient of first output with respect to second input is not zero: {first_gradients[1]}"
    assert torch.allclose(
        second_gradients[0], torch.zeros_like(second_gradients[0]), atol=ALL_CLOSE_ATOL
    ), f"Gradient of second output with respect to first input is not zero: {second_gradients[0]}"
    # Also verify that the gradient is not zero
    assert torch.any(
        first_gradients != 0
    ), f"Gradient of first output with respect to second input is all zeros: {first_gradients}"
    assert torch.any(
        second_gradients != 0
    ), f"Gradient of second output with respect to first input is all zeros: {second_gradients}"

    print("✅ Backward pass is batch independent.")


def assert_backward_causal_property(
    model_factory: ModelFactory,
    supervised_batch_provider: SupervisedBatchProvider,
    *,
    device: str = DEFAULT_DEVICE,
):
    """
    Asserts that the model's backward pass exhibits causal independence.

    This function checks if gradients for a token only depend on previous tokens,
    ensuring that the model maintains causal structure during backpropagation.

    Args:
        model_factory (ModelFactory): A function that returns a new instance of the model.
        supervised_batch_provider (SupervisedBatchProvider): The training data.
        device (str, optional): The device to move the tensors to.

    Raises:
        AssertionError: If the model's backward pass violates causal independence.
    """
    print("🔍 Checking backward causal property...")
    
    supervised_inputs, _ = get_supervised_batch(
        supervised_batch_provider, batch_size=1, device=device
    )
    supervised_inputs = supervised_inputs.clone().detach().requires_grad_(True)
    model = model_factory().to(device)
    replace_norm_layers_with_identity(model)

    model.train()
    model.zero_grad()

    # Forward pass
    outputs = model(supervised_inputs)

    sequence_length = outputs.shape[1]

    for t in range(sequence_length):
        # Compute gradient for output at time t
        if t > 0:
            supervised_inputs.grad.zero_()

        outputs[0, t].sum().backward(retain_graph=True)

        current_grad = supervised_inputs.grad.clone()

        # Check causal independence
        if t < sequence_length - 1:
            assert torch.allclose(
                current_grad[0, t + 1 :],
                torch.zeros_like(current_grad[0, t + 1 :]),
                atol=ALL_CLOSE_ATOL,
            ), f"Gradient at time {t} affects future tokens: {current_grad[0, t+1:]}"

        # Verify that the gradient for current and past tokens is not all zeros
        assert torch.any(
            current_grad[0, : t + 1] != 0
        ), f"Gradient for current and past tokens at time {t} is all zeros: {current_grad[0, :t+1]}"

    print("✅ Backward pass exhibits causal independence.")


def assert_input_independent_baseline_worse(
    model_factory: ModelFactory,
    supervised_batch_provider: SupervisedBatchProvider,
    *,
    train_batch_fn: train_batch = None,
    batch_size: int | None = None,
    loss_fn: torch.nn.Module = None,
    relative_loss_threshold: float = 0.1,
    device: str = DEFAULT_DEVICE,
    max_steps: int = 128,
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
        supervised_batch_provider (SupervisedBatchProvider): The training data.
        train_batch_fn (Callable, optional): A function to train the model for one batch.
        batch_size (int, optional): The batch size to use.
        loss_fn (torch.nn.Module, optional): The loss function to use.
        relative_loss_threshold (float, optional): The relative loss threshold for early stopping.
        device (str, optional): The device to move the tensors to.
        **kwargs: Additional keyword arguments to pass to train_batch_fn.

    Raises:
        AssertionError: If the model trained on real data doesn't outperform the one trained on fake data.
    """
    print("🔍 Checking input independence baseline worse...")
    
    if train_batch_fn is None:
        train_batch_fn = functools.partial(train_batch, **kwargs)
    else:
        assert kwargs == {}, "train_batch_fn and kwargs are mutually exclusive"
    if loss_fn is None:
        loss_fn = torch.nn.functional.cross_entropy

    regular_model = model_factory().to(device)
    supervised_batch = get_supervised_batch(
        supervised_batch_provider, batch_size=batch_size, device=device
    )

    input_independent_model = model_factory().to(device)
    fake_batch = (
        torch.zeros_like(supervised_batch[0]),
        supervised_batch[1],
    )
    for regular_loss_opt, _ in zip(
        train_batch_fn(
            regular_model,
            supervised_batch,
            loss_fn=loss_fn,
            device=device,
            num_steps=max_steps,
        ),
        train_batch_fn(
            input_independent_model,
            fake_batch,
            loss_fn=loss_fn,
            device=device,
            num_steps=max_steps,
        ),
    ):
        regular_loss, _ = regular_loss_opt
        input_independent_loss = loss_fn(
            input_independent_model(supervised_batch[0]),
            supervised_batch[1],
        )
        if (input_independent_loss - regular_loss) / max(
            regular_loss, input_independent_loss
        ) > relative_loss_threshold:
            break

    print(f"Regular loss: {regular_loss}")
    print(f"Input independent loss: {input_independent_loss}")
    assert (
        regular_loss < input_independent_loss
    ), f"Regular loss ({regular_loss}) is not less than input independent loss ({input_independent_loss})"
    
    print("✅ Input independent baseline is worse.")


def assert_overfit_to_batch(
    model: torch.nn.Module,
    supervised_batch_provider: SupervisedBatchProvider,
    *,
    train_batch_fn: train_batch = None,
    batch_size: int | None = None,
    threshold: float = 1e-4,
    max_steps: int = 2**16,
    optimizer: torch.optim.Optimizer | None = None,
    loss_fn: torch.nn.Module = None,
    device: str = DEFAULT_DEVICE,
    **kwargs,
):
    """
    Asserts that a model can overfit to a batch of data.

    This function trains a model on a batch of data and asserts that the loss decreases
    to a very small value (close to zero) within a reasonable number of steps.

    Args:
        model (torch.nn.Module): The model to train.
        supervised_batch_provider (SupervisedBatchProvider): The training data.
        train_batch_fn (Callable, optional): A function to train the model for one batch.
        batch_size (int, optional): The batch size to use.
        threshold (float, optional): The threshold for the loss to reach.
        max_steps (int, optional): The maximum number of steps to train for.
        optimizer (torch.optim.Optimizer, optional): The optimizer to use.
        loss_fn (torch.nn.Module, optional): The loss function to use.
        device (str, optional): The device to move the tensors to.
        **kwargs: Additional keyword arguments to pass to train_batch_fn.

    Raises:
        AssertionError: If the model fails to overfit to the batch.
    """
    print("🔍 Overfitting to the batch...")
    
    if train_batch_fn is None:
        train_batch_fn = functools.partial(train_batch, **kwargs)
    else:
        assert kwargs == {}, "train_batch_fn and kwargs are mutually exclusive"
    if loss_fn is None:
        loss_fn = torch.nn.functional.cross_entropy

    supervised_inputs, supervised_targets = get_supervised_batch(
        supervised_batch_provider, batch_size=batch_size, device=device
    )
    supervised_batch = (supervised_inputs, supervised_targets)
    model.to(device)

    loss = float("inf")
    for step, (loss, optimizer) in enumerate(
        train_batch_fn(
            model,
            supervised_batch,
            num_steps=max_steps,
            device=device,
            loss_fn=loss_fn,
        )
    ):
        if loss < threshold:
            print(f"Loss ({loss}) below threshold ({threshold}) at step {step}.")
            print("✅ Model can overfit to batch.")
            return loss, optimizer
    assert (
        False
    ), f"Failed to overfit to batch after {max_steps} steps. Final loss: {loss}."


def is_image_tensor(tensor):
    """
    Helper function to check if a tensor is an image tensor.

    Args:
        tensor (torch.Tensor): The tensor to check.

    Returns:
        bool: True if the tensor is an image tensor, False otherwise.
    """
    return len(tensor.shape) in [3, 4] and tensor.shape[len(tensor.shape) - 3] in [
        1,
        3,
        4,
    ]


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
            if tensor.dtype in [
                torch.uint8,
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            ]:
                if tensor.dtype == torch.uint8:
                    tensor = tensor.float() / 255.0
                    html += "<p>Note: Image data was not normalized for display.</p>"
                elif tensor.dtype in [
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                    torch.float64,
                ]:
                    tensor = tensor.float()
                    if tensor.max() > 1.0 or tensor.min() < 0.0:
                        min_val = tensor.min()
                        max_val = tensor.max()
                        tensor = (tensor - min_val) / (max_val - min_val)
                        html += f"<p>Note: Image data was normalized for display. Min: {min_val}, Max: {max_val}</p>"
                    else:
                        html += (
                            "<p>Note: Image data was not normalized for display.</p>"
                        )

                # Convert to PIL Images
                for image in tensor:
                    if image.shape[0] in [1, 3, 4]:
                        image = image.permute(1, 2, 0)
                    if image.shape[-1] == 1:
                        image = image.squeeze(-1)

                    img = Image.fromarray((image.numpy() * 255).astype("uint8"))

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
    print("🔧 Patching module to raise inputs right away...")
    module.forward = functools.wraps(module.forward)(raise_model_inputs)


def assert_all_for_classification_cross_entropy_loss(
    model_factory: ModelFactory,
    supervised_batch_provider: SupervisedBatchProvider,
    *,
    num_classes: int,
    device: str = DEFAULT_DEVICE,
):
    """
    Assert all conditions for a classification model with cross-entropy loss.

    Args:
        model_factory (ModelFactory): A function that returns a new instance of the model.
        supervised_batch_provider (SupervisedBatchProvider): The training data.
        num_classes (int): The number of classes in the dataset.
        device (str, optional): The device to move the tensors to.
    """
    print("🔍 Checking all conditions for classification model with cross-entropy loss...")
    assert_balanced_classification_cross_entropy_loss_at_init(
        model_factory(),
        supervised_batch_provider,
        num_classes=num_classes,
        device=device,
    )
    assert_balanced_classification_cross_entropy_init_calibrated(
        model_factory(),
        supervised_batch_provider,
        num_classes=num_classes,
        device=device,
    )
    assert_forward_batch_independence(
        model_factory, supervised_batch_provider, device=device
    )
    assert_non_zero_gradients(model_factory(), supervised_batch_provider, device=device)
    assert_backward_batch_independence(
        model_factory, supervised_batch_provider, device=device
    )
    assert_input_independent_baseline_worse(
        model_factory, supervised_batch_provider, device=device
    )
    assert_overfit_to_batch(model_factory(), supervised_batch_provider, device=device)
    print("✅ All conditions for classification model with cross-entropy loss verified.")

def replace_input_embedding_layer(
    model_factory: ModelFactory, embedding_layer_name: str
):
    """
    Replace the input embedding layer with a fixed embedding layer.
    """

    def fixed_model_factory():
        """
        A factory function that returns a new instance of the model
        with the input embedding layer replaced with a fixed embedding
        layer.
        """
        model = model_factory()
        # Find the embedding layer
        embedding_layer = model.get_submodule(embedding_layer_name)
        embedding_layer.forward = lambda x: x @ embedding_layer.weight
        return model

    return fixed_model_factory


def assert_all_for_causal_llm_cross_entropy_loss(
    model_factory: ModelFactory,
    supervised_batch_provider: SupervisedBatchProvider,
    *,
    embedding_layer_name: str | None = None,
    vocab_size: int,
    device: str = DEFAULT_DEVICE,
):
    """
    Assert all conditions for an LLM model with cross-entropy loss.

    Args:
        model_factory (ModelFactory): A function that returns a new instance of the model.
        supervised_batch_provider (SupervisedBatchProvider): The training data.
        vocab_size (int): The size of the vocabulary.
        device (str, optional): The device to move the tensors to.
    """
    print("🔍 Checking all conditions for a causal LLM model with cross-entropy loss...")
    
    if embedding_layer_name is not None:
        supervised_inputs, supervised_targets = get_supervised_batch(
            supervised_batch_provider, batch_size=min(128, vocab_size), device=device
        )
        # Convert the inputs into one-hot encoded vectors
        supervised_inputs = torch.nn.functional.one_hot(
            supervised_inputs, num_classes=vocab_size
        ).float()
        supervised_batch_provider = (supervised_inputs, supervised_targets)

        model_factory = replace_input_embedding_layer(
            model_factory, embedding_layer_name
        )

    def loss_fn(outputs, targets):
        assert outputs.shape[:-1] == targets.shape
        individual_loss = torch.nn.functional.cross_entropy(
            outputs.view(-1, vocab_size), targets.view(-1)
        )
        return individual_loss.mean()

    assert_balanced_classification_cross_entropy_loss_at_init(
        model_factory(),
        supervised_batch_provider,
        loss_fn=loss_fn,
        num_classes=vocab_size,
        device=device,
    )
    assert_balanced_classification_cross_entropy_init_calibrated(
        model_factory(),
        supervised_batch_provider,
        num_classes=vocab_size,
        device=device,
    )
    assert_forward_batch_independence(
        model_factory, supervised_batch_provider, device=device
    )
    assert_forward_causal_property(
        model_factory, supervised_batch_provider, device=device
    )
    assert_non_zero_gradients(model_factory(), supervised_batch_provider, device=device)
    assert_backward_batch_independence(
        model_factory, supervised_batch_provider, device=device
    )
    assert_backward_causal_property(
        model_factory, supervised_batch_provider, device=device
    )
    assert_input_independent_baseline_worse(
        model_factory, supervised_batch_provider, device=device, loss_fn=loss_fn
    )
    # For LLMs, we can overfit to a batch with a higher threshold
    assert_overfit_to_batch(
        model_factory(),
        supervised_batch_provider,
        device=device,
        loss_fn=loss_fn,
        threshold=1e-1,
    )
    print("✅ All conditions for a causal LLM model with cross-entropy loss verified.")