import torch.nn as nn
import torch.nn.functional as F

# approximate Q function using Neural Network
# state is input and Q Value of each action is output of network


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.V = nn.Linear(64, 1)
        self.A = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        V = self.V(x)
        A = self.A(x)
        return V, A


class DQN_Conv(nn.Module):

    def __init__(self, state_size, action_size, conv_size=(32, 64), fc_size=(1024, 128)):
        super(DQN_Conv, self).__init__()
        self.state_size = state_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, conv_size[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_size[0], conv_size[1],
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Linear(
            conv_size[1] * state_size * state_size, fc_size[0])
        self.fc2 = nn.Linear(fc_size[0], fc_size[1])
        self.V = nn.Linear(fc_size[1], 1)
        self.A = nn.Linear(fc_size[1], action_size)

    def forward(self, x):
        x = x.reshape(-1, 1, self.state_size, self.state_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        V = self.V(x)
        A = self.A(x)
        return V, A
