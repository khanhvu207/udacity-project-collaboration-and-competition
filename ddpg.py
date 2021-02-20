import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from memory import ReplayBuffer
from model import ActorNetwork, CriticNetwork
from config import *
from OUNoise import OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
	def __init__(self, state_size, action_size, seed):
		self.state_size = state_size
		self.action_size = action_size
		self.seed = random.seed(seed)

		# Actor networks
		self.actor_local = ActorNetwork(state_size, action_size, seed).to(device)
		self.actor_target = ActorNetwork(state_size, action_size, seed).to(device)
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR)

		# Critic networks
		self.critic_local = CriticNetwork(state_size, action_size, seed).to(device)
		self.critic_target = CriticNetwork(state_size, action_size, seed).to(device)
		self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR)

		# Replay memory
		self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, seed)

		# OUNoise
		self.exploration_noise = OUNoise(action_size, seed)

	def act(self, state):
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		
		self.actor_local.eval()
		with torch.no_grad():
			action = self.actor_local(state).cpu().data.numpy()
		self.actor_local.train()

		# Add exploration noise
		action += self.exploration_noise.sample()

		return np.clip(action, -1, 1)

	def step(self, state, action, reward, next_state, done):
		self.memory.add(state, action, reward, next_state, done)
		if len(self.memory) > BATCH_SIZE:
			self.learn()

	def learn(self):
		states, actions, rewards, next_states, dones = self.memory.sample()
		next_actions = self.actor_target(next_states)
		y = rewards + GAMMA * self.critic_target(next_states, next_actions) * (1 - dones)
		
		# Critic loss
		critic_loss = F.mse_loss(y, self.critic_local(states, actions))
		
		# Critic backprop 
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Actor loss
		cur_actions = self.actor_local(states)
		actor_loss = self.critic_local(states, cur_actions)
		actor_loss = -actor_loss.mean()

		# Actor backprop
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Soft updates
		self.update_target_network()

	def update_target_network(self):
		for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
			target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

		for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
			target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)