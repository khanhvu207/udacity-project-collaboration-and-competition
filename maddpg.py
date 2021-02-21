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

	def step(self, ep, state, action, reward, next_state, done):
		self.memory.add(state, action, reward, next_state, done)

		if len(self.memory) > BATCH_SIZE:
			for i in range(self.n_agents):
				self.learn(i)

	def learn(self, agent_index):
		states, actions, rewards, next_states, dones = self.memory.sample()
		
		target_next_actions = torch.from_numpy(np.zeros(shape=actions.shape)).float().to(device)
		for idx, agent in enumerate(self.ActorCriticAgents):
			current_states = states[:, idx]
			target_next_actions[:, idx, :] = agent.actor_target(current_states)

		target_next_actions = torch.reshape(target_next_actions, shape=(BATCH_SIZE, -1))

		current_agent_states = states[:, agent_index, :]
		current_agent_actions = actions[:, agent_index, :]
		current_agent_rewards = torch.reshape(rewards[:, agent_index], shape=(BATCH_SIZE, 1))
		current_agent_dones = torch.reshape(dones[:, agent_index], shape=(BATCH_SIZE, 1))

		action_preds = actions.clone()
		action_preds[:, agent_index, :] = self.ActorCriticAgents[agent_index].actor_local(current_agent_states)
		action_preds = torch.reshape(action_preds, shape=(BATCH_SIZE, -1))
		
		self.ActorCriticAgents[agent_index].update(states, current_agent_states, actions, current_agent_actions, target_next_actions, rewards, current_agent_rewards, next_states, dones, current_agent_dones, action_preds)
		
	def save_checkpoint(self):
		for i in range(self.n_agents):
			torch.save(self.ActorCriticAgents[i].actor_local.state_dict(), f'actor_checkpoint{i}.pth')
			torch.save(self.ActorCriticAgents[i].critic_local.state_dict(), f'critic_checkpoint{i}.pth')
