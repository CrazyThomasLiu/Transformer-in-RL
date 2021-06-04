import gym
env = gym.make('Pong-v0')
env.reset()
print(env.observation_space)
print(env.observation_space.shape)
print()
env.close()