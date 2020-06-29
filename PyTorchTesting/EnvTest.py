import gym

env = gym.make("Pong-ram-v0")
observation = env.reset()
for i in range(0,6):
    for _ in range(1000):
      env.render()
      action = env.action_space.sample() # your agent here (this takes random actions)
      action = i
      print(action)
      observation, reward, done, info = env.step(action)

      if done:
        observation = env.reset()
    env.close()
