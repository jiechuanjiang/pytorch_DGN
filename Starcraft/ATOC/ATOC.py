import math, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class ATT(nn.Module):
	def __init__(self, din):
		super(ATT, self).__init__()
		self.fc1 = nn.Linear(din, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 1)

	def forward(self, x):
		y = F.relu(self.fc1(x))
		y = F.relu(self.fc2(y))
		y = F.sigmoid(self.fc3(y))
		return y

class Encoder(nn.Module):
	def __init__(self, din=32, hidden_dim=128):
		super(Encoder, self).__init__()
		self.fc = nn.Linear(din, hidden_dim)

	def forward(self, x):
		embedding = F.relu(self.fc(x))
		return embedding

class CommModel(nn.Module):
	def __init__(self, n_node, din, hidden_dim, dout):
		super(CommModel, self).__init__()

		self.rnn=torch.nn.LSTM(input_size=din,hidden_size=hidden_dim,bidirectional=True,batch_first=True)
		self.fc = nn.Linear(hidden_dim*2, dout)
		self.n_node = n_node
		self.din = din
	def forward(self, x, mask):
		y = []
		for i in range(self.n_node):
			y.append(self.rnn(x*(mask[:,i].unsqueeze(-1).expand(-1,-1,self.din)))[0][:,i,:])
		y = torch.stack(y, dim=1)
		out = F.relu(self.fc(y))
		return out

class Q_Net(nn.Module):
	def __init__(self, hidden_dim, dout):
		super(Q_Net, self).__init__()
		self.fc = nn.Linear(hidden_dim, dout)

	def forward(self, x):
		q = self.fc(x)
		return q

class ATOC(nn.Module):
	def __init__(self,n_agent,num_inputs,hidden_dim,num_actions):
		super(ATOC, self).__init__()
		
		self.encoder = Encoder(num_inputs,hidden_dim)
		self.comm_1 = CommModel(n_agent,hidden_dim,hidden_dim,hidden_dim)
		self.comm_2 = CommModel(n_agent,hidden_dim,hidden_dim,hidden_dim)
		self.q_net = Q_Net(hidden_dim,num_actions)
		
	def forward(self, x, mask):
		h1 = self.encoder(x)
		h2 = self.comm_1(h1, mask)
		#h3 = self.comm_2(h2, mask)
		q = self.q_net(h2)
		return q 















