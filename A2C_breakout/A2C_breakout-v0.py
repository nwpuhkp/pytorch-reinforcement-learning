import os.path
import time

import torch.nn.utils as tutils
import gym
import numpy as np
import torch
from skimage.transform import resize
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical

num_episodes = 1000
max_timesteps = 200
log_interval = 100
save_interval = 100
gamma = 0.99
save_path = 'model/breakout.pth'
eps = np.finfo(np.float32).eps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(state):
    return resize(state[35:195].mean(2), (80, 80), mode='reflect') / 255.0#图片处理

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.flat_dimension = 32 * 5 * 5
        self.hidden_dimension = 512

        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.rnn = nn.GRU(self.flat_dimension,self.hidden_dimension,batch_first=True)#直接用GRU网络

        self.action_brain = nn.Sequential(
            nn.Linear(self.hidden_dimension,2000),
            nn.ReLU(),
            nn.Linear(2000,200),
            nn.ReLU(),
            nn.Linear(200,4)
        )

        self.value_brain = nn.Sequential(
            nn.Linear(self.hidden_dimension,2000),
            nn.ReLU(),
            nn.Linear(2000, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self,x,hx = None):
        x = x.view(-1,1,80,80)
        x = self.cnn(x)
        x = x.view(-1,1,self.flat_dimension)
        x,hx = self.rnn(x,hx)

        action_probs = F.log_softmax(self.action_brain(hx).squeeze(),dim=0)
        state_value = self.value_brain(hx).squeeze()

        return action_probs,state_value,hx


class BreakoutAgent:#训练以及保存模型
    def __init__(self,model = None):
        self.model = model if model is not None else Policy()
        self.optimizer = optim.SGD(self.model.parameters(),lr=1e-3)

        self.log_probs = []
        self.entropy = []
        self.values = []
        self.rewards = []
        self.hx = None
        self.gamma = 0.99
        self.model.to(device)

    def getAction(self,env,state):
        state = preprocess(state)
        state = torch.from_numpy(state).float().to(device)
        probs,state_value,self.hx = self.model(state,self.hx)

        m = Categorical(logits=probs.squeeze())#按照传入的probs中给定的概率，在相应的位置处进行取样，取样返回的是该位置的整数索引
        action = m.sample()
        log_prob = m.log_prob(action)

        self.log_probs.append(log_prob)
        self.entropy.append(m.entropy())
        self.values.append(state_value)

        return probs,action.item()

    def feed(self,reward):
        self.rewards.append(reward)

    def learn(self):
        loss = train_agent(self,self.gamma,device)
        del self.log_probs[:] #结束后清空
        del self.rewards[:]
        del self.values[:]
        del self.entropy[:]
        self.hx = None
        return loss

    def load(self,path):
        self.model.load_state_dict(torch.load(path))#加载已有模型
    def save(self,path):
        torch.save(self.model.state_dict(),path)#保存训练模型
    def train(self):
        self.model.train()#训练模式
    def eval(self):
        self.model.eval()#测试

def GetTime(start_time):
    take_time = int(time.time() - start_time)
    s = int(take_time % 60)
    take_time /= 60
    m = int(take_time % 60)
    h = int(take_time/60)

    return '{:02d} h {:02d} m {:02d} s'.format(h,m,s)

def DiscountedReturns(rewards,gamma):
    R = 0.0
    returns = []
    for r in rewards[::-1]:#更新R
        R = r + gamma * R
        returns.insert(0,R)
    return returns


def train_agent(agent, gamma, device=None):
    returns = DiscountedReturns(agent.rewards,gamma)
    returns = torch.tensor(returns)
    log_probs = torch.stack(agent.log_probs)
    values = torch.stack(agent.values)
    entropy = torch.stack(agent.entropy)

    if device is not None:
        returns = returns.to(device)
        log_probs = log_probs.to(device)
        values = values.to(device)

    advantage = returns - values
    actor_loss = (-log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2.0).mean()
    entropy_loss = entropy.mean()
    loss = actor_loss + critic_loss - 0.01 * entropy_loss

    agent.optimizer.zero_grad()
    loss.backward()
    tutils.clip_grad_norm_(agent.model.parameters(), 40.0)
    agent.optimizer.step()
    return loss.item()

class States:
    #每个情节的状态
    def __init__(self):
        self.avg_rewards = 0.0
        self.avg_timesteps = 0.0
        self.avg_loss = 0.0
        self.start_time = time.time()
        self.episodes = 0
        self.last_reward = 0.0
        self.last_timesteps = 0.0
        self.last_loss = 0.0
        self.max_reward = 0.0
    def update(self,reward,timesteps,loss):
        self.episodes += 1
        self.avg_rewards += (reward-self.avg_rewards)/self.episodes
        self.avg_timesteps += (timesteps - self.avg_timesteps)/self.episodes
        self.avg_loss += (loss - self.avg_loss)/self.episodes
        self.last_reward = reward
        self.last_timesteps = timesteps
        self.last_loss = loss
        if self.max_reward<reward:
            self.max_reward = reward

    def reset(self):#重置
        self.avg_rewards = 0.0
        self.avg_timesteps = 0.0
        self.avg_loss = 0.0
        self.episodes = 0
        self.last_reward = 0.0
        self.last_timesteps = 0.0
        self.last_loss = 0.0
        self.max_reward = 0.0

    def printstates(self):
        print('{}, Total Episodes: {}'.format(GetTime(self.start_time), self.episodes))
        print('Max Reward: {}'.format(self.max_reward))
        print('avg Reward:    {:10.03f}, Last Reward:    {:10.03f}'.format(self.avg_rewards, self.last_reward))
        print('avg Timesteps: {:10.03f}, Last Timesteps: {:10.03f}'.format(self.avg_timesteps, self.last_timesteps))
        print('avg Loss:      {:10.03f}, Last Loss:      {:10.03f}'.format(self.avg_loss, self.last_loss))


def trainmodel(model):
    env = gym.make('Breakout-v0')
    start_time = time.time()
    agent = BreakoutAgent(model)
    state = States()
    last_lives = 5.0

    agent.train()
    for episode in range(1,num_episodes+1):
        current_state = env.reset()#开始游戏获得状态
        ep_reward = 0.0

        for t in range(1,max_timesteps+1):
            probs,action = agent.getAction(env,current_state)
            current_state,reward,done,info= env.step(action)

            if done:
                reward = -1.0

            agent.feed(reward)
            ep_reward +=reward

            if done:
                break
        loss = agent.learn()
        state.update(ep_reward,t,loss)

        if (episode%log_interval) == 0:
            # print(probs)
            state.printstates()
            state.reset()

        if (episode%save_interval) == 0:
            agent.save(save_path)

    env.close()

if __name__ =='__main__':
    agent = BreakoutAgent()
    if os.path.exists(save_path):
        agent.load(save_path)
    trainmodel(agent.model)