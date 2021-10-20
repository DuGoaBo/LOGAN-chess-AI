import torch.nn as nn

class LOGAN_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.conv3 = nn.Conv2d(8, 16, 3)
        self.conv4 = nn.Conv2d(16, 32, 2)
        self.flatten = nn.Flatten(1, 3)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.flatten(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    L = LOGAN_module()
    import torch
    x = torch.Tensor(1, 1, 8, 8)
    L(x)
