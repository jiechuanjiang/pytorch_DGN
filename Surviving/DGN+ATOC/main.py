import math, random, copy
import numpy as np
import os,sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from model import DGN, ATT
from buffer import ReplayBuffer
from surviving import Surviving
from config import *

USE_CUDA = torch.cuda.is_available()

env = Surviving(n_agent = 100)
n_ant = env.n_agent
observation_space = env.len_obs
n_actions = env.n_action

buff = ReplayBuffer(capacity,observation_space,n_actions,n_ant)
model = DGN(n_ant,observation_space,hidden_dim,n_actions)
model_tar = DGN(n_ant,observation_space,hidden_dim,n_actions)
model = model.cuda()
model_tar = model_tar.cuda()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
att = ATT(observation_space).cuda()
att_tar = ATT(observation_space).cuda()
att_tar.load_state_dict(att.state_dict())
optimizer_att = optim.Adam(att.parameters(), lr = 0.0001)
criterion = nn.BCELoss()

M_Null = torch.Tensor(np.array([np.eye(n_ant)]*batch_size)).cuda()
M_ZERO = torch.Tensor(np.zeros((batch_size,n_ant,n_ant))).cuda()
threshold = float(sys.argv[1])
f = open(sys.argv[1]+'-'+sys.argv[2]+'.txt','w')
while i_episode<n_episode:

	if i_episode > 40:
		epsilon -= 0.001
		if epsilon < 0.01:
			epsilon = 0.01
	i_episode+=1
	steps = 0
	obs, adj = env.reset()
	while steps < max_step:
		steps+=1 
		action=[]
		cost_all += adj.sum()
		v_a = np.array(att(torch.Tensor(np.array([obs])).cuda())[0].cpu().data)
		for i in range(n_ant):
			if np.random.rand() < epsilon:
				adj[i] = adj[i]*0 if np.random.rand() < 0.5 else adj[i]*1
			else:
				adj[i] = adj[i]*0 if v_a[i][0] < threshold else adj[i]*1
		n_adj = adj*comm_flag
		cost_comm += n_adj.sum()
		n_adj = n_adj + np.eye(n_ant)
		q = model(torch.Tensor(np.array([obs])).cuda(), torch.Tensor(np.array([n_adj])).cuda())[0]
		for i in range(n_ant):
			if np.random.rand() < epsilon:
				a = np.random.randint(n_actions)
			else:
				a = q[i].argmax().item()
			action.append(a)

		next_obs, next_adj, reward, terminated = env.step(action)

		buff.add(np.array(obs),action,reward,np.array(next_obs),n_adj,next_adj,terminated)
		obs = next_obs
		adj = next_adj
		score += sum(reward)

	if i_episode%20==0:
		f.write(str(score/2000)+'	'+str(cost_comm/cost_all)+'\n')
		f.flush()
		score = 0
		cost_comm = 0
		cost_all = 0

	if i_episode < 40:
		continue

	for e in range(n_epoch):
		
		O,A,R,Next_O,Matrix,Next_Matrix,D = buff.getBatch(batch_size)
		O = torch.Tensor(O).cuda()
		Matrix = torch.Tensor(Matrix).cuda()
		Next_O = torch.Tensor(Next_O).cuda()
		Next_Matrix = torch.Tensor(Next_Matrix).cuda()

		label = model(Next_O, Next_Matrix+M_Null).max(dim = 2)[0] - model(Next_O, M_Null).max(dim = 2)[0]
		label = (label - label.mean())/(label.std()+0.000001) + 0.5
		label = torch.clamp(label, 0, 1).unsqueeze(-1).detach()
		loss = criterion(att(Next_O), label)
		optimizer_att.zero_grad()
		loss.backward()
		optimizer_att.step()

		V_A_D = att_tar(Next_O).expand(-1,-1,n_ant)
		Next_Matrix = torch.where(V_A_D > threshold, Next_Matrix, M_ZERO)
		Next_Matrix = Next_Matrix*comm_flag + M_Null

		q_values = model(O, Matrix)
		target_q_values = model_tar(Next_O, Next_Matrix).max(dim = 2)[0]
		target_q_values = np.array(target_q_values.cpu().data)
		expected_q = np.array(q_values.cpu().data)
		
		for j in range(batch_size):
			for i in range(n_ant):
				expected_q[j][i][A[j][i]] = R[j][i] + (1-D[j])*GAMMA*target_q_values[j][i]
		
		loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		with torch.no_grad():
			for p, p_targ in zip(model.parameters(), model_tar.parameters()):
				p_targ.data.mul_(tau)
				p_targ.data.add_((1 - tau) * p.data)
			for p, p_targ in zip(att.parameters(), att_tar.parameters()):
				p_targ.data.mul_(tau)
				p_targ.data.add_((1 - tau) * p.data)
		