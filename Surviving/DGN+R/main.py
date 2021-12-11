import math, random, copy
import numpy as np
import os,sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from model import DGN
from buffer import ReplayBuffer
from surviving import Surviving
from config import *

USE_CUDA = torch.cuda.is_available()

lamb = float(sys.argv[1])

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

M_Null = torch.Tensor(np.array([np.eye(n_ant)]*batch_size)).cuda()
KL = nn.KLDivLoss()
f = open(sys.argv[1]+'_'+sys.argv[2]+'.txt','w')
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
		n_adj = adj + np.eye(n_ant)
		q, a_w = model(torch.Tensor(np.array([obs])).cuda(), torch.Tensor(np.array([n_adj])).cuda())
		q = q[0]
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
		f.write(str(score/2000)+'\n')
		f.flush()
		score = 0

	if i_episode < 40:
		continue

	for e in range(n_epoch):
		
		O,A,R,Next_O,Matrix,Next_Matrix,D = buff.getBatch(batch_size)
		O = torch.Tensor(O).cuda()
		Matrix = torch.Tensor(Matrix).cuda()
		Next_O = torch.Tensor(Next_O).cuda()
		Next_Matrix = torch.Tensor(Next_Matrix).cuda()
		Next_Matrix = Next_Matrix + M_Null

		q_values, attention = model(O, Matrix)
		target_q_values, target_attention = model_tar(Next_O, Next_Matrix)
		target_q_values = target_q_values.max(dim = 2)[0]
		target_q_values = np.array(target_q_values.cpu().data)
		expected_q = np.array(q_values.cpu().data)
		
		for j in range(batch_size):
			for i in range(n_ant):
				expected_q[j][i][A[j][i]] = R[j][i] + (1-D[j])*GAMMA*target_q_values[j][i]
		

		attention = F.log_softmax(attention,dim=2)
		target_attention = F.softmax(target_attention,dim=2)
		target_attention = target_attention.detach()
		loss_kl = F.kl_div(attention, target_attention, reduction='mean')

		loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean() + lamb*loss_kl
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		with torch.no_grad():
			for p, p_targ in zip(model.parameters(), model_tar.parameters()):
				p_targ.data.mul_(tau)
				p_targ.data.add_((1 - tau) * p.data)
		