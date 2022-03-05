
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from numpy import mean
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from env import *

# hyper-parameters
BATCH_SIZE = 30
LR = 0.001
GAMMA = 0.9
EPISILO = 0.9
EPS_END = 0.002
MEMORY_CAPACITY = 800
Q_NETWORK_ITERATION = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = mfcc_env()

NUM_ACTIONS = env.action_space_dim
NUM_STATES = env.state_space_dim
ENV_A_SHAPE = 0

def make_state(state):
    state = state[:19]
    return state

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES,256)
        # self.fc1.weight.data.normal_(0,0.1)
        # self.softmax1 = nn.Softmax(dim=1)
        self.fc2 = nn.Linear(256,256)
        # self.softmax2 = nn.Softmax(dim=1)
        self.fc3 = nn.Linear(256,NUM_ACTIONS)
        # self.fc2.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.fc3(x)
        return action_prob

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net().to(device), Net().to(device)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss().to(device)
        self.loss_num = 0

    def choose_action(self, state,eps):
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(device) # get a 1D array
        # state = torch.FloatTensor(state).to(device)
        if np.random.randn() > eps:# greedy policy
            action_value = self.eval_net.forward(state).to(device)
            action = torch.max(action_value, 1)[1].data.cpu().numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            # print(action)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
            # print(action)
        return action


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [[action, reward]], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES]).to(device)
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int)).to(device)
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2]).to(device)
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:]).to(device)

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action).to(device)
        q_next = self.target_net(batch_next_state).detach().to(device)
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target).to(device)
        self.loss_num += loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



if __name__ == '__main__':
    dqn = DQN()
    eps = EPISILO
    eps_end = EPS_END
    eps_decay = 0.95
    episodes = 1500
    print("Collecting Experience....")
    plt.figure(figsize=(12,3))
    plt.ion()
    plot_x1_data, plot_y1_data = [], []
    plot_y2_data = []
    for i in range(episodes):
        state,index= env.reset()
        state = make_state(state)
        ep_reward = 0
        # print(state)
        dqn.loss_num = 0
        while True:
            action = dqn.choose_action(state,eps)
            next_state, reward, done, index = env.step(action,index)
            next_state = make_state(next_state)
            print(state)
            print(reward)
            print(action)
            print(next_state)

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            # print('ep_reward: ', ep_reward, 'reward: ', reward)

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {} loss is {}".format(i, round(ep_reward, 3),round(dqn.loss_num.item(),2)))
            if done:
                break
            state = next_state
        eps = max(eps_end, eps_decay * eps)

        plot_x1_data.append(i)
        plot_y1_data.append(ep_reward)
        # plot_y2_data.append(dqn.loss_num)
        plt.subplot(1, 1, 1).set_title('reward')
        plt.plot(plot_x1_data, plot_y1_data,'b-')
        # plt.subplot(1, 2, 2).set_title('loss')
        # plt.plot(plot_x1_data, plot_y2_data,'r-')


        plt.pause(0.1)
        if i != 0 and i % 500 == 0:
            torch.save(dqn, './model/500.pkl')
    plt.ioff()
    plt.show()


