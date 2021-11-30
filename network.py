import torch
from torch import nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    
    def __init__(self, in_features, action_shape):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu = nn.Linear(64,action_shape[0])
        # use .5 as initial std -> log(.5) ~= -0.3
        self.log_std = nn.Parameter(torch.ones(action_shape) * -0.3, requires_grad=True)
    
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return self.mu(x), self.log_std

class ValueNet(nn.Module):
    
    def __init__(self, in_features):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.v_head = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        v = self.v_head(x)
        return v