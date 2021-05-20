import os, sys  
import numpy as np
from smac.env import StarCraft2Env
from DGN import DGN
from buffer import ReplayBuffer
from config import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
env = StarCraft2Env(map_name='8m')#'25m'
env_info = env.get_env_info()
n_ant = env_info["n_agents"]
n_actions = env_info["n_actions"]
obs_space = env_info["obs_shape"]

buff = ReplayBuffer(capacity,obs_space,n_actions,n_ant)
model = DGN(n_ant,obs_space,hidden_dim,n_actions)
model_tar = DGN(n_ant,obs_space,hidden_dim,n_actions)
model = model.cuda()
model_tar = model_tar.cuda()
model_tar.load_state_dict(model.state_dict())
optimizer = optim.RMSprop(model.parameters(), lr = 0.0005)

while i_episode<n_episode:

	if i_episode > 40:
		epsilon -= 0.005
		if epsilon < 0.05:
			epsilon = 0.05

	i_episode+=1
	steps = 0
	env.reset()
	terminated = False
	episode_reward = 0
	win = 0
	obs = env.get_obs()
	adj = env.get_visibility_matrix()[:,0:n_ant]*1 + np.eye(n_ant)
	mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
	while not terminated:
		action=[]
		q = model(torch.Tensor(np.array([obs])).cuda(), torch.Tensor(adj).cuda())[0]
		for i in range(n_ant):
			if np.random.rand() < epsilon:
				avail_actions_ind = np.nonzero(mask[i])[0]
				a = np.random.choice(avail_actions_ind)
			else:
				a = np.argmax(q[i].cpu().detach().numpy() - 9e15*(1 - mask[i]))
			action.append(a)
		reward, terminated, winner = env.step(action)
		if winner.get('battle_won') == True:
			win = 1
		episode_reward += reward
		next_obs = env.get_obs()
		next_adj = env.get_visibility_matrix()[:,0:n_ant]*1 + np.eye(n_ant)
		mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
		buff.add(np.array(obs),action,reward,np.array(next_obs),adj,next_adj,mask,terminated)
		obs = next_obs
		adj = next_adj

	sum_reward += episode_reward
	sum_win += win
	print("Total reward in episode {} = {}".format(i_episode, episode_reward))
	if i_episode%200 == 0:
		print(str(i_episode)+'	'+str(sum_win/200)+'	'+str(sum_reward/200))
		print(sum_win/200)
		print(sum_reward/200)
		sum_reward = 0
		sum_win = 0
	
	if i_episode < 40:
		continue

	for epoch in range(n_epoch):
		
		O,A,R,Next_O,Matrix,Next_Matrix,Next_Mask,D = buff.getBatch(batch_size)

		q_values = model(torch.Tensor(O).cuda(), torch.Tensor(Matrix).cuda())
		target_q_values = model_tar(torch.Tensor(Next_O).cuda(), torch.Tensor(Next_Matrix).cuda())
		target_q_values = (target_q_values - 9e15*(1 - torch.Tensor(Next_Mask).cuda())).max(dim = 2)[0]
		target_q_values = np.array(target_q_values.cpu().data)
		expected_q = np.array(q_values.cpu().data)
		
		for j in range(batch_size):
			for i in range(n_ant):
				expected_q[j][i][A[j][i]] = R[j] + (1-D[j])*GAMMA*target_q_values[j][i]
		
		loss = (q_values - torch.Tensor(expected_q).cuda()).pow(2).mean()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if i_episode%5 == 0:
		model_tar.load_state_dict(model.state_dict())

