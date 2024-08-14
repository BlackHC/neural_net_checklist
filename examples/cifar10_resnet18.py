# %%
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# %%
# Define ResNet18 architecture
def ResNet18(num_classes=10):
    model = torchvision.models.resnet18(
        pretrained=False, num_classes=num_classes, norm_layer=nn.Identity
    )
    # Modify the first convolutional layer to match CIFAR-10 input size
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove the max pooling layer
    model.maxpool = nn.Identity()
    # assert_all_for_classification_cross_entropy_loss failed, zero out bias
    model.fc.bias.data.zero_()

    return model


# %%
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 dataset
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_dataset = torchvision.datasets.CIFAR10(
    root="../data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="../data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# %%
import neural_net_checklist.torch_recipe as torch_recipe

torch_recipe.ModelInputs(torch_recipe.get_supervised_data(train_loader))

# %%
torch_recipe.assert_all_for_classification_cross_entropy_loss(
    ResNet18, train_loader, 10
)

# %%
