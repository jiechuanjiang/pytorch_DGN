import numpy as np
import torch
def test_agent(test_env, model, att, n_ant, comm_flag, threshold):

	test_r, test_win = 0, 0
	for _ in range(20):
		test_env.reset()
		test_obs = test_env.get_obs()
		test_adj = test_env.get_visibility_matrix()[:,0:n_ant]*1
		test_mask = np.array([test_env.get_avail_agent_actions(i) for i in range(n_ant)])
		terminated = False
		while terminated == False:
			action=[]
			v_a = att(torch.Tensor(np.array([test_obs])).cuda())[0]
			for i in range(n_ant):
				if v_a[i][0] < threshold:
					test_adj[i] = test_adj[i]*0
			test_n_adj = test_adj*comm_flag+np.eye(n_ant)
			q = model(torch.Tensor(np.array([test_obs])).cuda(), torch.Tensor(np.array([test_n_adj])).cuda())[0]
			for i in range(n_ant):
				a = np.argmax(q[i].cpu().detach().numpy() - 9e15*(1 - test_mask[i]))
				action.append(a)
			reward, terminated, winner = test_env.step(action)
			if winner.get('battle_won') == True:
				test_win += 1
			test_r += reward
			test_obs = test_env.get_obs()
			test_adj = test_env.get_visibility_matrix()[:,0:n_ant]*1 
			test_mask = np.array([test_env.get_avail_agent_actions(i) for i in range(n_ant)])
		
	return test_r/20, test_win/20
