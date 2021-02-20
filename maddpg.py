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

class MADDPG():
	def __init__(self, state_size, action_size, n_agents, seed):

	def act(self, state):
	
	def step(self, state, action, reward, next_state, done):
	
	def learn(self):