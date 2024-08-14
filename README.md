# neural_net_checklist: A codified recipe for training neural networks

This repository is based on Andrei Karpathy's well-known blog post "A recipe for training neural networks" (https://karpathy.github.io/2019/04/25/recipe/). Instead of being a checklist, this repository implements a set of assertions that can be used to diagnose neural networks. 

This is useful to save time when debugging neural networks without having to think of each assertion anew every time.

Specifically, this repository implements:

- "verify loss @ init" for classification tasks with balanced classes -> `assert_balanced_classification_cross_entropy_loss_at_init`
- "init well" for classification tasks with balanced classes -> `assert_balanced_classification_cross_entropy_loss_at_init`
- non-zero gradients -> `assert_non_zero_gradients`
- batch independence:
  - using a forward pass (uses less memory) --- NOTE: batchnorm breaks this naturally! -> `assert_batch_independence_forward`
  - ("use backprop to chart dependencies") using a backward pass (uses more memory) and checking the gradients -> `assert_batch_independence_backward`
- "overfit one batch" -> `assert_overfit_one_batch`
- "visualize just before the net" -> `patch_module_raise_inputs`
