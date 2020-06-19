import argparse
import gym
import numpy as np
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


seed = 6969

env = gym.make('Breakout-ram-v0')
env.seed(seed)
torch.manual_seed(seed)


render = False
gamma = 1
log_interval = 1

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.fc1 = nn.Linear(128, 128)
        self.d1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(128,128)
        self.d2 = nn.Dropout(p=0.5)
       
        self.fc3 = nn.Linear(128,128)
        self.d3 = nn.Dropout(p=0.5)
       
        self.fc4 = nn.Linear(128,128)
        self.d4 = nn.Dropout(p=0.5)
        
        self.fc5 = nn.Linear(128,128)
        self.d5 = nn.Dropout(p=0.5)
        
        self.action_layer = nn.Linear(128,4)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.d1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = self.d2(x)
        x = torch.relu(x)
        
        x = self.fc3(x)
        x= self.d3(x)
        x = torch.relu(x)
        
        x = self.fc4(x)
        x = self.d4(x)
        x = torch.relu(x)
        
        x = self.fc5(x)
        x = self.d5(x)
        x = torch.relu(x)

        action_scores = self.action_layer(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-4)
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
    running_reward = 1
    et_max = 0

    reward_list = list()

    for i_episode in range(0,1000):

        env.seed(i_episode)
        torch.manual_seed(i_episode)

        state, ep_reward = env.reset(), 0
        et_reward = 1

        for t in range(1, 2000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            
            if done and et_reward == 1:
                reward = -2
            elif done:
                reward = -1
            
            if render:
                env.render()

            policy.rewards.append(reward)
            ep_reward += reward
            et_reward += reward

            if done:
               #env.reset()
               #print(et_reward)
               if et_reward > et_max:
                    et_max = et_reward
               et_reward = 0
               break
                   
        reward_list.append(et_reward);
        et_reward = 0

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()

        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tMax reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward, et_max))

    fig,ax = plt.subplots()
    ax.plot(reward_list)
    ax.grid()
    plt.show()

if __name__ == '__main__':
    main()

