import string

import torch.nn as nn
import torch.nn.functional as F

ALPHABET = string.digits + string.ascii_uppercase + string.ascii_lowercase

class CharacterClassifier(nn.Module):
    """Convolution neural network to recognize typed characters."""
    def __init__(self, num_classes=len(ALPHABET), F=3, M=5):
        super().__init__()
        # w_out = (w_in - F + 2P)/S + 1
        # input is (3,128,128) image
        # For ease of use, always use enough padding s.t w_out = w_in
        P = (F-1)//2
        self.name = f'nc={num_classes}:F={F}:M={M}'
        self.conv1 = nn.Conv2d(3, M, kernel_size=F, stride=1, padding=P)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(M, 5, kernel_size=F, stride=1, padding=P)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(5*32*32, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # (3,128,128) -> (5,64,64)
        x = self.pool1(F.relu(self.conv1(x)))
        # (5,64,64) -> (5,32,32)
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 5*32*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x