import gym
env = gym.make('BreakoutNoFrameskip-v4')
env.reset()
#print(env.observation_space)
#print(env.observation_space.shape)
#print(env.observation_space.shape[0])
print(env.action_space)
#print(env.action_space.shape)
env.close()