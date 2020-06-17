import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


seed = 6969

env = gym.make('Breakout-ram-v0')
env.seed(seed)
torch.manual_seed(seed)

render = True
gamma = 0.99
log_interval = 1

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.fc1 = nn.Linear(128, 128)

        
        self.fc2 = nn.Linear(128, 100)
        self.d1 = nn.Dropout(p=0.1)

        self.fc3 = nn.Linear(100, 100)
        self.d2 = nn.Dropout(p=0.1)

        self.fc4 = nn.Linear(100, 50)
        self.d3 = nn.Dropout(p=0.1)

        self.fc5 = nn.Linear(50, 4)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)

        x = self.fc2(x)
        x = self.d1(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = self.d2(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = self.d3(x)
        x = F.relu(x)

        action_scores = self.fc5(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state):

    state = torch.from_numpy(state.flatten()).float()
    state = state.view(-1,128)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):

        env.seed(i_episode)

        state, ep_reward = env.reset(), 0
        et_reward = 0
        for t in range(1, 1000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            
            if et_reward>=3:
                reward = reward*3
            else:
                reward = reward*0.1
            reward = reward*10
            
            if done and et_reward == 0:
                reward = -10

            if render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            et_reward += reward
            if done:
               env.reset()
               print(et_reward)
               et_reward = 0

               #break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()

        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        #if running_reward > env.spec.reward_threshold:
        #    print("Solved! Running reward is now {} and "
        #          "the last episode runs to {} time steps!".format(running_reward, t))
        #    break


if __name__ == '__main__':
    main()

