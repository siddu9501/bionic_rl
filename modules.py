import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np 
from collections import deque
import random

class ReplayBuffer:

	def __init__(self, cf):
		self.buffer_size = cf.max_buffer
		self.len = 0

		#Create buffers 
		self.buffer = deque(maxlen=self.buffer_size)

	def sample(self, count):
		count = min(count, self.len)
		batch = random.sample(self.buffer, count)

		s_t, a_t, r_t, s_tp1 = zip(*batch)
		s_t = Variable(torch.from_numpy(np.asarray(s_t)))
		a_t = Variable(torch.from_numpy(np.asarray(a_t)))
		r_t = Variable(torch.from_numpy(np.asarray(r_t)))
		s_tp1 = Variable(torch.from_numpy(np.asarray(s_tp1)))

		return s_t, a_t, r_t, s_tp1

	def add(self, s_t, a_t, r_t, s_tp1):
		transition = (s_t, a_t, r_t, s_tp1)
		self.len += 1
		if self.len > self.buffer_size:
			self.len = self.buffer_size
		self.buffer.append(transition)

class OrnsteinUhlenbeckNoise():
	def __init__(self, cf):
		self.action_dim = cf.action_dim
		self.mu = cf.mu
		self.theta = cf.theta
		self.sigma = cf.sigma
		self.X = np.ones(self.action_dim) * self.mu

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		self.X += self.theta * (self.mu - self.X) + self.sigma * np.random.randn(self.action_dim)
		return self.X

class Actor(nn.Module):
	def __init__(self, cf):
		super(Actor, self).__init__()
		self.model = nn.Sequential(
			nn.Linear(cf.state_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128,64),
			nn.ReLU(),
			nn.Linear(64, cf.action_dim),
			nn.Tanh()
			)
		for i in [0, 2, 4]:
			nn.init.xavier_uniform(self.model[i].weight.data)
		self.model[-2].weight.data.uniform_(-3e-3, 3e-3)

	def forward(self, state):
		return self.model(state)

class Critic(nn.Module):
	def __init__(self, cf):
		super(Critic, self).__init__()
		self.transform_state = nn.Sequential(
			nn.Linear(cf.state_dim, 128),
			nn.ReLU()
			)
		nn.init.xavier_uniform(self.transform_state[0].weight.data)

		self.transform_action = nn.Sequential(
			nn.Linear(cf.action_dim, 128),
			nn.ReLU()
			)
		nn.init.xavier_uniform(self.transform_action[0].weight.data)

		self.transform_both = nn.Sequential(
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128,1)
			)
		nn.init.xavier_uniform(self.transform_both[0].weight.data)
		self.transform_both[-1].weight.data.uniform_(-3e-3, 3e-3)


	def forward(self, state, action):
		state = self.transform_state(state)
		action = self.transform_action(action)
		both = torch.cat([state, action], 1)
		return self.transform_both(both)


class DDPG(nn.Module):
	def __init__(self, cf):
		super(DDPG, self).__init__()
		self.cf = cf

		self.actor = Actor(cf)
		self.actor_optimizer = optim.Adam(self.actor.parameters(),
			cf.actor_learning_rate)
		self.actor_target = Actor(cf)

		self.critic = Critic(cf)
		self.critic_optimizer = optim.Adam(self.critic.parameters(),
			cf.critic_learning_rate)
		self.critic_target = Critic(cf)

		self.buffer = ReplayBuffer(cf)
		self.noise = OrnsteinUhlenbeckNoise(cf)


	def update_targets(self):
		for actor, actor_target in zip(self.actor.parameters(), self.actor_target.parameters()):
			actor_target.data.copy_(self.cf.tau*actor.data + (1-self.cf.tau)*actor_target.data)

		for critic, critic_target in zip(self.critic.parameters(), self.critic_target.parameters()):
			critic_target.data.copy_(self.cf.tau*critic.data + (1-self.cf.tau)*critic_target.data)

	
	def copy_weights(self):
		for actor, actor_target in zip(self.actor.parameters(), self.actor_target.parameters()):
			actor_target.data.copy_(actor.data)

		for critic, critic_target in zip(self.critic.parameters(),self.critic_target.parameters()):
			critic_target.data.copy_(critic.data)
	def copy_weights1(self):
		for actor, actor_target in zip(self.actor.parameters(), self.actor_target.parameters()):
			actor.data.copy_(actor_target.data)

		for critic, critic_target in zip(self.critic.parameters(),self.critic_target.parameters()):
			critic.data.copy_(critic_target.data)


	def sample_action(self, state, explore=True):
		state = Variable(torch.from_numpy(state))
		action = self.actor(state[None]).cpu().data.numpy()
		if explore:
			action = action + self.noise.sample()
		return action

	def train_batch(self):
		s_t, a_t, r_t, s_tp1 = self.buffer.sample(self.cf.batch_size)

		a_tp1 = self.actor_target(s_tp1).detach()
		q_value = self.critic_target(s_tp1, a_tp1).squeeze().detach()
		y_target = r_t + self.cf.gamma*q_value
		y_predicted = self.critic(s_t, a_t).squeeze()

		critic_loss = F.smooth_l1_loss(y_predicted, y_target)
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		a_t_pred = self.actor(s_t)
		q_pred = self.critic(s_t, a_t_pred)
		actor_loss = -1*torch.mean(q_pred)
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self.update_targets()
		return critic_loss, actor_loss

	def save_models(self):
		torch.save(self.actor_target.state_dict(), 'models/best_actor.model')
		torch.save(self.critic_target.state_dict(), 'models/best_critic.model')

	def load_models(self):
		self.actor_target.load_state_dict(
			torch.load('models_old/best_actor.model'))
		self.critic_target.load_state_dict(
			torch.load('models_old/best_critic.model'))
