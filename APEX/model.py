import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.A = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        V = self.V(x)
        A = self.A(x)
        return V, A
