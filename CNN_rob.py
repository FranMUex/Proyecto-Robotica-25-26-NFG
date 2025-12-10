import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def show_pytorch_mosaic(rows=10, cols=10):
    """
    Loads MNIST using PyTorch and displays a mosaic using make_grid.
    """
    # 1. Define Transformations
    # PyTorch requires converting images to Tensors (0-1 range)
    transform = transforms.Compose([transforms.ToTensor()])

    # 2. Load the Dataset
    # download=True checks if data exists; if not, downloads it to ./data
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

    # 3. Create a DataLoader
    # We use a batch_size equal to the total images we want to show
    num_images = rows * cols
    dataloader = torch.utils.data.DataLoader(trainset, batch_size=num_images, shuffle=True)

    # 4. Get a single batch of images
    # dataiter returns (images, labels)
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # 5. Create the Grid
    # make_grid stitches the batch into a single image tensor
    # padding=0 ensures a seamless "mosaic" look without borders
    mosaic_tensor = torchvision.utils.make_grid(images, nrow=cols, padding=0)

    # 6. Prepare for Plotting
    # PyTorch images are (Channels, Height, Width) -> (1, 280, 280)
    # Matplotlib expects (Height, Width, Channels) -> (280, 280, 1)
    # We use .permute() to rearrange dimensions
    mosaic_np = mosaic_tensor.permute(1, 2, 0).numpy()

    # 7. Display
    plt.figure(figsize=(10, 10))
    plt.imshow(mosaic_np, cmap='gray_r') # Inverted grayscale
    plt.axis('off')
    plt.title(f"PyTorch MNIST Mosaic ({rows}x{cols})")
    plt.show()
    
    print(f"Displayed mosaic of {num_images} images.")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])

    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1000, shuffle=False
    )

    model = torch.load("my_network.pt", map_location=device)
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f" Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    test()
    #show_pytorch_mosaic(rows=12, cols=12)

    