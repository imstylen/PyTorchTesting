import gym
env = gym.make("Breakout-v0")
for _ in range(100):
    observation = env.reset()
    for _ in range(500):
      env.render()
      action = env.action_space.sample() # your agent here (this takes random actions)
      observation, reward, done, info = env.step(action)

