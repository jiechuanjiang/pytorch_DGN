import os, sys  
import numpy as np
from smac.env import StarCraft2Env
from ATOC import ATOC, ATT
from buffer import ReplayBuffer
from config import *
from utilis import *
import torch
import torch.nn as nn
import torch.optim as optim 
threshold = float(sys.argv[1])
env = StarCraft2Env(map_name='10m_vs_11m',range_=4.0)#4.0
env_info = env.get_env_info()
n_ant = env_info["n_agents"]
n_actions = env_info["n_actions"]
obs_space = env_info["obs_shape"]

buff = ReplayBuffer(capacity,obs_space,n_actions,n_ant)
model = ATOC(n_ant,obs_space,hidden_dim,n_actions)
model_tar = ATOC(n_ant,obs_space,hidden_dim,n_actions)
model = model.cuda()
model_tar = model_tar.cuda()
model_tar.load_state_dict(model.state_dict())
optimizer = optim.RMSprop(model.parameters(), lr = 0.0005)
att = ATT(obs_space).cuda()
att_tar = ATT(obs_space).cuda()
att_tar.load_state_dict(att.state_dict())
optimizer_att = optim.RMSprop(att.parameters(), lr = 0.0005)
criterion = nn.BCELoss()

M_Null = torch.Tensor(np.array([np.eye(n_ant)]*batch_size)).cuda()
M_ZERO = torch.Tensor(np.zeros((batch_size,n_ant,n_ant))).cuda()

f = open(sys.argv[1]+'_'+sys.argv[2]+'.txt','w')
while i_episode<n_episode:
	if time_step > 250000:
		break
	if i_episode > 100:
		epsilon -= 0.001
		if epsilon < 0.02:
			epsilon = 0.02

	i_episode+=1
	env.reset()
	terminated = False
	obs = env.get_obs()
	adj = env.get_visibility_matrix()[:,0:n_ant]*1 #+ np.eye(n_ant)
	mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
	while not terminated:
		test_flag += 1
		time_step += 1
		action=[]
		v_a = np.array(att(torch.Tensor(np.array([obs])).cuda())[0].cpu().data)
		for i in range(n_ant):
			if np.random.rand() < epsilon:
				adj[i] = adj[i]*0 if np.random.rand() < 0.5 else adj[i]*1
			else:
				adj[i] = adj[i]*0 if v_a[i][0] < threshold else adj[i]*1
		n_adj = adj*comm_flag + np.eye(n_ant)
		q = model(torch.Tensor(np.array([obs])).cuda(), torch.Tensor(np.array([n_adj])).cuda())[0]
		for i in range(n_ant):
			if np.random.rand() < epsilon:
				avail_actions_ind = np.nonzero(mask[i])[0]
				a = np.random.choice(avail_actions_ind)
			else:
				a = np.argmax(q[i].cpu().detach().numpy() - 9e15*(1 - mask[i]))
			action.append(a)
		reward, terminated, winner = env.step(action)
		next_obs = env.get_obs()
		next_adj = env.get_visibility_matrix()[:,0:n_ant]*1
		next_mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
		buff.add(np.array(obs),action,reward,np.array(next_obs),n_adj,next_adj,next_mask,terminated,mask)
		obs = next_obs
		adj = next_adj
		mask = next_mask
	if test_flag > 10000:
		log_r, log_w, log_cost = test_agent(env, model, att, n_ant, comm_flag, threshold)
		h = str(log_r)+'	'+str(log_w)+'	'+str(log_cost)
		f.write(h+'\n')
		f.flush()
		test_flag = 0
	
	if i_episode < 100:
		continue

	for epoch in range(n_epoch):
		
		O,A,R,Next_O,Matrix,Next_Matrix,Next_Mask,D,Mask = buff.getBatch(batch_size)
		O = torch.Tensor(O).cuda()
		Matrix = torch.Tensor(Matrix).cuda()
		Next_O = torch.Tensor(Next_O).cuda()
		Next_Matrix = torch.Tensor(Next_Matrix).cuda()
		Mask = torch.Tensor(Mask).cuda()
		Next_Mask = torch.Tensor(Next_Mask).cuda()
		
		label = (model(Next_O, Next_Matrix+M_Null) - 9e15*(1 - Next_Mask)).max(dim = 2)[0] - (model(Next_O, M_Null) - 9e15*(1 - Next_Mask)).max(dim = 2)[0]
		##optional
		#mu = label.mean().data
		#std = label.std().data
		#label = torch.clamp(label, mu-std, mu+std)
		label = (label - label.mean())/(label.std()+0.000001) + 0.5
		label = torch.clamp(label, 0, 1).unsqueeze(-1).detach()
		loss = criterion(att(Next_O), label)
		optimizer_att.zero_grad()
		loss.backward()
		optimizer_att.step()

		#optional
		#V_A_D = att_tar(Next_O).expand(-1,-1,n_ant)
		#Next_Matrix = torch.where(V_A_D > threshold, Next_Matrix, M_ZERO)
		Next_Matrix = Next_Matrix*comm_flag + M_Null

		q_values = model(O, Matrix)
		target_q_values = model_tar(Next_O, Next_Matrix)
		target_q_values = (target_q_values - 9e15*(1 - Next_Mask)).max(dim = 2)[0]
		target_q_values = np.array(target_q_values.cpu().data)
		expected_q = np.array(q_values.cpu().data)
		
		for j in range(batch_size):
			for i in range(n_ant):
				expected_q[j][i][A[j][i]] = R[j] + (1-D[j])*GAMMA*target_q_values[j][i]
		
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

