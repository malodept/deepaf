
import torch, torch.nn as nn

class FocusDiscriminatorTiny(nn.Module):
    def __init__(self):
        super().__init__()
        c = 16
        self.net = nn.Sequential(
            nn.Conv2d(3, c, 5, 2, 2), nn.ReLU(True),
            nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(c, 2*c, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(2*c, 2*c, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(2*c, 4*c, 3, 2, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4*c, 64), nn.ReLU(True),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)
