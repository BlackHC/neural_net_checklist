import torch
import torch.optim
import typing
import numpy as np
import functools

from tqdm.auto import tqdm


# A dummy input is either a (batched) tensor or a function that takes a batch size and returns a tensor of that batch size.
DummySupervisedBatch = (
    tuple[torch.Tensor, torch.Tensor]
    | typing.Callable[[int | None], tuple[torch.Tensor, torch.Tensor]]
)


ModelFactory = typing.Callable[[], torch.nn.Module]


def get_dummy_supervised_data(
    dummy_batch: DummySupervisedBatch, batch_size: int | None
):
    if isinstance(dummy_batch, tuple):
        if batch_size is None:
            batch_size = dummy_batch[0].shape[0]

        assert (
            dummy_batch[0].shape[0] >= batch_size
        ), f"Dummy batch only has {dummy_batch[0].shape[0]} samples, but batch size is {batch_size}"
        return dummy_batch[0][:batch_size], dummy_batch[1][:batch_size]
    else:
        return dummy_batch(batch_size)


def train_batch(
    model: torch.nn.Module,
    dummy_batch: DummySupervisedBatch,
    optimizer: torch.optim.Optimizer | None = None,
    num_steps: int = 512,
    batch_size: int | None = None,
):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    inputs, targets = get_dummy_supervised_data(dummy_batch, batch_size)

    model.train()
    for _ in tqdm(range(num_steps)):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

    return loss.item()


def assert_balanced_classification_cross_entropy_loss_at_init(
    model: torch.nn.Module, dummy_batch: DummySupervisedBatch, num_classes: int
):
    """
    Asserts that the loss at initialization is close to the expected loss for a balanced classification problem.

    This function checks if the initial loss of the model is approximately equal to -log(1/num_classes),
    which is the expected loss when the model's predictions are uniformly distributed across all classes.
    A small tolerance is allowed to account for minor variations due to random initialization.
    """
    dummy_inputs, dummy_targets = get_dummy_supervised_data(dummy_batch, num_classes)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(dummy_inputs)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(outputs, dummy_targets)

    expected_loss = np.log(num_classes)
    tolerance = 0.5

    print(
        f"Loss at initialization ({loss.item():.4f}) is within expected range ({expected_loss - tolerance:.4f}, {expected_loss + tolerance:.4f})"
    )
    assert (
        abs(loss.item() - expected_loss) <= tolerance
    ), f"Loss at initialization ({loss.item():.4f}) is not within expected range ({expected_loss - tolerance:.4f}, {expected_loss + tolerance:.4f})"


def assert_balanced_classification_crossentropy_init_calibrated(
    model: torch.nn.Module,
    dummy_batch: DummySupervisedBatch,
    num_classes: int,
    batch_size: int = 512,
):
    """
    Asserts that the model is well-calibrated at initialization, i.e. the model's predictions are uniformly distributed across all classes.
    """
    dummy_inputs, _ = get_dummy_supervised_data(
        dummy_batch, batch_size
    )  # Larger batch for better statistics
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        outputs = model(dummy_inputs)
        probs = torch.softmax(outputs, dim=-1)

        # Calculate the expected probability (1/num_classes for balanced dataset)
        expected_prob = 1.0 / num_classes
        # Check if all class probabilities are close to the expected probability
        tolerance = 0.5
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
    model: torch.nn.Module, dummy_batch: DummySupervisedBatch, train_mode: bool = True
):
    """
    Asserts that the model's forward pass is invariant to the order of the inputs in the batch.
    """
    # Check that if we take 3 samples  [A, B, C] then:
    # the results for [A], [B], [C] == [A, B, C]
    # the results for [A, B] == [A, B]
    # the results for [B, C] == [B, C]
    # etc.
    dummy_inputs, _ = get_dummy_supervised_data(dummy_batch, 3)
    if train_mode:
        model.train()
    else:
        model.eval()

    with torch.no_grad():
        a, b, c = dummy_inputs.unbind(0)
        ab_out = model(torch.cat([a[None, :], b[None, :]], dim=0))
        bc_out = model(torch.cat([b[None, :], c[None, :]], dim=0))
        ac_out = model(torch.cat([a[None, :], c[None, :]], dim=0))
        abc_out = model(torch.cat([a[None, :], b[None, :], c[None, :]], dim=0))

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
    model: torch.nn.Module, dummy_batch: DummySupervisedBatch
):
    """
    Asserts that the model's gradients are all non-zero.
    """
    dummy_inputs, dummy_targets = get_dummy_supervised_data(dummy_batch, 1)
    model.train()
    model.zero_grad()

    outputs = model(dummy_inputs)
    outputs.sum().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert torch.any(param.grad != 0), f"Gradient for {name} is all zeros"

    print("All gradients are non-zero.")


def assert_backward_batch_independence(
    model: torch.nn.Module, dummy_batch: DummySupervisedBatch
):
    """
    Asserts that the model's backward pass is invariant to the order of the inputs in the batch.
    """
    # We can simply take two samples and make sure that if we take the gradient for the second input in regard to the first outputs
    # it is zero.
    dummy_inputs, _ = get_dummy_supervised_data(dummy_batch, 2)
    dummy_inputs.requires_grad_(True)

    model.train()
    model.zero_grad()

    # Forward pass
    outputs = model(dummy_inputs)

    # Calculate gradients for the first output with respect to inputs
    first_output = outputs[0].sum()
    first_output.backward(retain_graph=True)

    # Store gradients for the first output
    first_gradients = dummy_inputs.grad.clone()

    # Zero out the gradients
    dummy_inputs.grad.zero_()

    # Calculate gradients for the second output with respect to inputs
    second_output = outputs[1].sum()
    second_output.backward()

    # Store gradients for the second output
    second_gradients = dummy_inputs.grad.clone()

    # Verify that the other gradient is zero each time
    assert torch.allclose(
        first_gradients[1], torch.zeros_like(first_gradients[1])
    ), "Gradient of first output with respect to second input is not zero"
    assert torch.allclose(
        second_gradients[0], torch.zeros_like(second_gradients[0])
    ), "Gradient of second output with respect to first input is not zero"
    # Also verify that the gradient is not zero
    assert torch.any(
        first_gradients != 0
    ), "Gradient of first output with respect to second input is all zeros"
    assert torch.any(
        second_gradients != 0
    ), "Gradient of second output with respect to first input is all zeros"

    print("Backward pass is batch independent.")


def assert_input_independence_baseline_worse(model_factory: ModelFactory, dummy_batch: DummySupervisedBatch, train_batch_fn: typing.Callable[[], torch.nn.Module] = None, batch_size: int | None = None, **kwargs):
    if train_batch_fn is None:
        train_batch_fn = functools.partial(train_batch, dummy_batch=dummy_batch, **kwargs)
    else:
        assert kwargs == {}, "train_batch_fn and kwargs are mutually exclusive"
    
    regular_model = model_factory()
    dummy_batch = get_dummy_supervised_data(dummy_batch, batch_size)
    train_batch(regular_model, dummy_batch)
    
    input_independent_model = model_factory()
    fake_batch = (torch.zeros_like(dummy_batch[0]), dummy_batch[1])
    train_batch(input_independent_model, fake_batch)
    
    # Evaluate both models on dummy batch
    loss = torch.nn.CrossEntropyLoss()
    regular_loss = loss(regular_model(dummy_batch[0]), dummy_batch[1])
    input_independent_loss = loss(input_independent_model(dummy_batch[0]), dummy_batch[1])
    
    print(f"Regular loss: {regular_loss}")
    print(f"Input independent loss: {input_independent_loss}")
    assert regular_loss < input_independent_loss, f"Regular loss ({regular_loss}) is not less than input independent loss ({input_independent_loss})"