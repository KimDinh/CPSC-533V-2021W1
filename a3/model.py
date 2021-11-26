import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(MyModel, self).__init__()
        # Basic Multilayer Perceptron
        self.fc1 = nn.Linear(state_size,64)
        self.rl1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64,64)
        self.rl2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        return self.fc3(self.rl2(self.fc2(self.rl1(self.fc1(x)))))

    def select_action(self, state):
        self.eval()
        x = self.forward(state)
        self.train()
        return x.max(1)[1].view(1, 1).to(torch.long)
