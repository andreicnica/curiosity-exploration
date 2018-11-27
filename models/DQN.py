import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, cfg, input_size, action_size):
        super(DQN, self).__init__()
        self.activation = activation = F.relu
        self.hidden_size = hidden_size = cfg.hidden_size

        self.use_batch_norm = use_batch_norm = False

        def null_(x):
            return x

        self.conv1 = nn.Conv2d(input_size[0], 32, kernel_size=4, stride=1)
        self.bn1 = nn.BatchNorm2d(16) if use_batch_norm else null_
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batch_norm else null_
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32) if use_batch_norm else null_

        self.ln1 = nn.Linear(32*25*25, hidden_size)
        self.bnln1 = nn.BatchNorm1d(hidden_size) if use_batch_norm else null_
        self.head = nn.Linear(hidden_size, action_size)

    def forward(self, x, states=None, masks=None):
        act = self.activation
        x = act(self.bn1(self.conv1(x)))
        x = act(self.bn2(self.conv2(x)))
        x = act(self.bn3(self.conv3(x)))
        x = act(self.bnln1(self.ln1(x.view(x.size(0), -1))))
        return self.head(x)

