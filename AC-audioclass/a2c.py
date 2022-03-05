
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from env import *
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = mfcc_env()

state_size = env.state_space_dim
action_size = env.action_space_dim
# print(state_size,action_size)
lr = 0.0001
gamma = 0.99
episodes = 2000
def make_state(state):
    state = state[:19]
    return state

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.dnn1 = nn.Linear(32,32)
        self.dnn2 = nn.Linear(32,3)


    def forward(self, x):
        x = x.view(-1,1,19,12)
        x = self.cnn(x)
        x = x.view(-1, 1, 32)
        x = F.relu(self.dnn1(x))
        x = self.dnn2(x)
        # print(x)
        x = x.view(-1,1)
        # print(x)
        distribution = Categorical(F.softmax(x.squeeze(), dim=-1))
        # print(distribution)
        return distribution



class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.dnn1 = nn.Linear(32, 32)
        self.dnn2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(-1, 1, 19, 12)
        x = self.cnn(x)
        x = x.view(-1, 1, 32)
        x = F.relu(self.dnn1(x))
        x = self.dnn2(x)
        return x


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def train(actor, critic, episodes):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    plt.figure(figsize=(12,3))
    plt.ion()
    plot_x1_data, plot_y1_data = [], []
    plot_y2_data = []


    for i_episode in range(episodes):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        state,index = env.reset()
        # print(index)
        state = make_state(state)
        ep_rewards = 0

        while True:
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)
            # print(dist)

            action = dist.sample()
            next_state, reward, done, index = env.step(action.cpu().numpy(),index)
            next_state = make_state(next_state)
            ep_rewards += reward

            log_prob = dist.log_prob(action).unsqueeze(0)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state



            if done:
                print('episodes_number: {}, Score: {}'.format(i_episode, ep_rewards))
                break


        plot_x1_data.append(i_episode)
        plot_y1_data.append(ep_rewards)
        # plot_y2_data.append(dqn.loss_num)
        plt.subplot(1, 1, 1).set_title('reward')
        plt.plot(plot_x1_data, plot_y1_data,'b-')
        # plt.subplot(1, 2, 2).set_title('loss')
        # plt.plot(plot_x1_data, plot_y2_data,'r-')


        plt.pause(0.1)


        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    
    plt.ioff()
    plt.show()




if __name__ == '__main__':
    actor = Actor(state_size, action_size).to(device)
    critic = Critic(state_size, action_size).to(device)
    train(actor, critic, episodes)
