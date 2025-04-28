import torch
import torch.nn as nn

class myCNNModel(torch.nn.Module):
    def __init__(self):
        super(myCNNModel, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        
        # Convolutional Layer 2
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
        
        # Compute the size of the flattened feature map (for 28x28 input)
        # After conv1 (26x26), pool1 (13x13), conv2 (11x11), pool2 (5x5)
        self.fc1 = torch.nn.Linear(in_features=32 * 5 * 5, out_features=800)
        self.fc2 = torch.nn.Linear(in_features=800, out_features=10)

    def forward(self, x):
        x = torch.torch.nn.functionalrelu(self.conv1(x))   # Conv1 + ReLU
        x = self.pool1(x)           # Max Pool1
        x = torch.torch.nn.functionalrelu(self.conv2(x))   # Conv2 + ReLU
        x = self.pool2(x)           # Max Pool2
        x = x.view(-1, 32 * 5 * 5)  # Flatten
        x = torch.torch.nn.functionalrelu(self.fc1(x))     # Fully connected layer
        x = self.fc2(x)             # Output layer
        return x
