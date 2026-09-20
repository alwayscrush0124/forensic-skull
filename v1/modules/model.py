import torch, torch.nn as nn
class SkullMultiTaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d(2),
            nn.AdaptiveAvgPool3d((4, 4, 4))
        )
        self.gender_head = nn.Sequential(nn.Flatten(), nn.Linear(32*4*4*4, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x):
        features = self.encoder(x)
        return self.gender_head(features), None
