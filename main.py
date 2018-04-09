import gym
import gc
from config import load_config
from modules import DDPG
import numpy as np 


cf = load_config('config/baseline.py')
env = gym.make('BipedalWalker-v2')

cf.state_dim = env.observation_space.shape[0]
cf.action_dim = env.action_space.shape[0]

print 'State Dimensions :-', cf.state_dim
print 'Action Dimensions :-', cf.action_dim

model = DDPG(cf)
model.load_models()
model.copy_weights()

losses = []

model.noise.reset()
for epi in range(cf.max_episodes):
	s_t = env.reset().astype('float32')
	print 'Episode :- ', epi
	for r in range(cf.max_steps):
		env.render()
		a_t = model.sample_action(s_t, epi%10==0).flatten()
		s_tp1, r_t, done, info = env.step(a_t)

		if done:
			new_state = None
		else:
			s_tp1 = s_tp1.astype('float32')
			r_t = r_t.astype('float32')
			a_t = a_t.astype('float32')
			model.buffer.add(s_t, a_t, r_t, s_tp1)

		s_t = s_tp1
		_loss_c, _loss_a = model.train_batch()
		losses.append([_loss_c.cpu().data.tolist()[0], _loss_a.cpu().data.tolist()[0]])

		if done:
			break

	gc.collect()

	print "Episode {}: actor loss: {} critic loss: {}".format(
        epi, np.mean(np.asarray(losses), 0)[1],
        np.mean(np.asarray(losses), 0)[0])

	model.save_models()

print "Completed Episodes"

