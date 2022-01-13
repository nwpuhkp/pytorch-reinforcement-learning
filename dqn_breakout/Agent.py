import random

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from Qnet import QNetwork
from ReplayMemory import ReplayMemory

MEMORY_SIZE = int(1e6) #初始化大小
BATCH_SIZE = 32        #单次传递给程序用以训练的参数个数
GAMMA = 0.99
TAU = 1e-3             #这里采用DDPG中的soft update
LR = 1e-5              #learning rate
UPDATE_EVERY = 4       #多久更新一次网络

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

class Agent():
    #与环境交互
    def __init__(self,state_size,action_size,seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        #Qnet
        self.qnet_local = QNetwork(state_size,action_size,seed).to(device)#动作行为
        self.qnet_target = QNetwork(state_size,action_size,seed).to(device)#target
        self.optimizer = optim.Adam(self.qnet_local.parameters(),lr=LR)#优化

        #ReplayMemory
        self.memory = ReplayMemory(action_size,MEMORY_SIZE,BATCH_SIZE,seed)
        self.t_step = 0#初始化步长，用于更新UPDATE_EVERY steps

    def step(self,state,action,reward,next_state,done):
        #往replaymemory中保存experience
        self.memory.add(state,action,reward,next_state,done)

        #每隔设置的时间学习
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            #如果memory中的样例足够拿来训练，就从中取出并学习
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences,GAMMA)

    def act(self,state,eps=0):#epslion
        #根据当前策略返回给定状态的动作。
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state)
        self.qnet_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())#选value最大的动作
        else:
            return random.choice(np.arange(self.action_size))#随机选取一个动作


    def learn(self,experiences,gamma):
        #更新Q table，v值
        states,actions,rewards,next_states,dones = experiences
        #损失函数
        #从目标模型中获得最大的预测Q值(对于next_state)
        Q_targets_next = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)#截断反向传播的梯度流
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnet_local(states).gather(1,actions)#固定行号，确认列号

        #计算损失
        loss = F.mse_loss(Q_expected,Q_targets).to(device)

        #最小化loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #更新网络
        self.soft_update(self.qnet_local,self.qnet_target,TAU)


    def soft_update(self,local_model,target_model,tau):
        #DDPG 公式θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)





