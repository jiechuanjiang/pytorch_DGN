import os, sys  
import numpy as np
from smac.env import StarCraft2Env
from model import DGN
from buffer import ReplayBuffer
from config import *
from utilis import *
import torch
import torch.optim as optim
env = StarCraft2Env(map_name='25m',range_=range_)
env_info = env.get_env_info()
n_ant = env_info["n_agents"]
n_actions = env_info["n_actions"]
obs_space = env_info["obs_shape"] + n_ant

buff = ReplayBuffer(capacity,obs_space,n_actions,n_ant)
model = DGN(n_ant,obs_space,hidden_dim,n_actions)
model_tar = DGN(n_ant,obs_space,hidden_dim,n_actions)
model = model.cuda()
model_tar = model_tar.cuda()
model_tar.load_state_dict(model.state_dict())
optimizer = optim.RMSprop(model.parameters(), lr = 0.0005)
comm_flag = float(sys.argv[1])

f = open(sys.argv[1]+'_'+sys.argv[2]+'.txt','w')
while i_episode<n_episode:
	if time_step > max_timestep:
		break
	if i_episode > 100:
		epsilon -= 0.001
		if epsilon < 0.02:
			epsilon = 0.02
	i_episode+=1
	env.reset()
	terminated = False
	obs = get_obs(env.get_obs(),n_ant)
	adj = env.get_visibility_matrix()[:,0:n_ant]*comm_flag + np.eye(n_ant)
	mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
	while not terminated:
		test_flag += 1
		time_step += 1

		action=[]
		q = model(torch.Tensor(np.array([obs])).cuda(), torch.Tensor(np.array([adj])).cuda())[0]
		for i in range(n_ant):
			if np.random.rand() < epsilon:
				avail_actions_ind = np.nonzero(mask[i])[0]
				a = np.random.choice(avail_actions_ind)
			else:
				a = np.argmax(q[i].cpu().detach().numpy() - 9e15*(1 - mask[i]))
			action.append(a)
			
		reward, terminated, winner = env.step(action)
		next_obs = get_obs(env.get_obs(),n_ant)
		next_adj = env.get_visibility_matrix()[:,0:n_ant]*comm_flag + np.eye(n_ant)
		mask = np.array([env.get_avail_agent_actions(i) for i in range(n_ant)])
		buff.add(np.array(obs),action,reward,np.array(next_obs),adj,next_adj,mask,terminated)
		obs = next_obs
		adj = next_adj

	if test_flag > 10000:
		log_r, log_w = test_agent(env, model, n_ant)
		h = str(log_r)+'	'+str(log_w)
		f.write(h+'\n')
		f.flush()
		test_flag = 0
	
	if i_episode < 100:
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

		with torch.no_grad():
			for p, p_targ in zip(model.parameters(), model_tar.parameters()):
				p_targ.data.mul_(tau)
				p_targ.data.add_((1 - tau) * p.data)

