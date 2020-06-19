import gym
env = gym.make("LunarLander-v2")
for _ in range(100):
    observation = env.reset()
    print((env.action_space))
    print(len(observation))
    for _ in range(500):
      env.render()
      action = env.action_space.sample() # your agent here (this takes random actions)
      observation, reward, done, info = env.step(action)

