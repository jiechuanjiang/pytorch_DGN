import numpy as np
import torch
def get_obs(obs,n_agent):
	for i in range(n_agent):
		index = np.zeros(n_agent)
		index[i] = 1
		obs[i] = np.hstack((obs[i],index))
	return obs
def test_agent(test_env, model, n_ant):

	test_r, test_win = 0, 0
	for _ in range(20):
		test_env.reset()
		test_obs = get_obs(test_env.get_obs(),n_ant)
		test_adj = test_env.get_visibility_matrix()[:,0:n_ant]*1 + np.eye(n_ant)
		test_mask = np.array([test_env.get_avail_agent_actions(i) for i in range(n_ant)])
		terminated = False
		while terminated == False:
			action=[]
			q = model(torch.Tensor(np.array([test_obs])).cuda(), torch.Tensor(np.array([test_adj])).cuda())[0]
			for i in range(n_ant):
				a = np.argmax(q[i].cpu().detach().numpy() - 9e15*(1 - test_mask[i]))
				action.append(a)
			reward, terminated, winner = test_env.step(action)
			test_r += reward
			if winner.get('battle_won') == True:
				test_win += 1
			test_obs = get_obs(test_env.get_obs(),n_ant)
			test_adj = test_env.get_visibility_matrix()[:,0:n_ant]*1 + np.eye(n_ant)
			test_mask = np.array([test_env.get_avail_agent_actions(i) for i in range(n_ant)])
	return test_r/20, test_win/20
