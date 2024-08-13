# A codified recipe for training neural networks

This repository is based on Andrei Karpathy's famous blog post "A recipe for training neural networks" (https://karpathy.github.io/2019/04/25/recipe/).

Specifically, this repository implements:

- "verify loss @ init" for classification tasks with balanced classes
- "init well" for classification tasks with balanced classes
- non-zero gradients
- batch independence:
  - using a forward pass (uses less memory) --- NOTE: batchnorm breaks this naturally!
  - ("use backprop to chart dependencies") using a backward pass (uses more memory) and checking the gradients 
- 