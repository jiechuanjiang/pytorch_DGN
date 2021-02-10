from collections import deque 
import random
class ReplayBuffer(object):

	def __init__(self, buffer_size):
		self.buffer_size = buffer_size
		self.num_experiences = 0
		self.buffer = deque()

	def getBatch(self, batch_size):
		if self.num_experiences < batch_size:
			return random.sample(self.buffer, self.num_experiences)
		else:
			return random.sample(self.buffer, batch_size)

	def add(self, obs, action, reward, new_obs, matrix, next_matrix, done):
		experience = (obs, action, reward, new_obs, matrix, next_matrix, done)
		if self.num_experiences < self.buffer_size:
			self.buffer.append(experience)
			self.num_experiences += 1
		else:
			self.buffer.popleft()
			self.buffer.append(experience)