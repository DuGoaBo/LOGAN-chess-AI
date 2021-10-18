import torch.nn as nn

class LOGAN_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(8*8 + 3, 120)
        self.h2 = nn.Linear(120, 80)
        self.y = nn.Linear(80, 1)

    def forward(self, x):
        leaky_relu = nn.LeakyReLU(0.1)
        x = leaky_relu(self.h1(x))
        x = leaky_relu(self.h2(x))
        x = self.y(x)
        return x

if __name__ == "__main__":
    L = LOGAN_module()
    import torch
    print(L(torch.ones(67)))
