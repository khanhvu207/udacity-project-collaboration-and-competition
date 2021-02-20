import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from memory import ReplayBuffer
from config import *
from ddpg import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
	def __init__(self, state_size, action_size, n_agents, seed):
		self.state_size = state_size
		self.action_size = action_size
		self.n_agents = n_agents
		self.seed = random.seed(seed)
		
		# Actor-Critic agents
		self.ActorCriticAgents = [Agent(state_size, action_size, n_agents, seed) for _ in range(n_agents)]
		
		# Replay memory
		self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, seed)

	def OUNoise_reset(self):
		for agent in self.ActorCriticAgents:
			agent.exploration_noise.reset()

	def act(self, state):
		actions = []
		for i, agent in enumerate(self.ActorCriticAgents):
			agent_action = agent.act(state[i])
			actions.append(agent_action[0])
		return np.stack(actions, axis=0)

	def step(self, state, action, reward, next_state, done):
		return 0

	def learn(self):
		return 0