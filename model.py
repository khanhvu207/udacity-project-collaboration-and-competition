import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
	def __init__(self, state_size, action_size, seed, hidden_size1=128, hidden_size2=128):
		super(ActorNetwork, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size, hidden_size1)
		self.fc2 = nn.Linear(hidden_size1, hidden_size2)
		self.fc3 = nn.Linear(hidden_size2, action_size)
		self.bn1 = nn.BatchNorm1d(hidden_size1)

	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = self.bn1(x)
		x = F.relu(self.fc2(x))
		x = F.tanh(self.fc3(x))
		return x

class CriticNetwork(nn.Module):
	def __init__(self, state_size, action_size, seed, hidden_size1=128, hidden_size2=128):
		super(CriticNetwork, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size, hidden_size1)
		self.fc2 = nn.Linear(hidden_size1 + action_size, hidden_size2)
		self.fc3 = nn.Linear(hidden_size2, 1)
		self.bn1 = nn.BatchNorm1d(hidden_size1)

	def forward(self, state, action):
		x = F.relu(self.fc1(state))
		x = self.bn1(x)
		x = torch.cat([x, action], dim=1)
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x