import gym
import torch
from torch.distributions import Categorical

from PG_train import Policy

model = Policy()
model.load_state_dict(torch.load('hkp_PG_train4.pth'))
model.eval()#非训练模式

def selet_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = model(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item()

env = gym.make('CartPole-v1')
t_all = []
for i_episode in range(2):
    observation = env.reset()
    for t in range(10000):
        env.render()
        cp,cv,pa,pv = observation
        action = selet_action(observation)
        observation,reward,done,info = env.step(action)
        if done:#倒了或者连续运行了500次都会done
            #print("Episode finished after {} timesteps".format(t+1))
            # print("倒了")
            print(t)#499表示完成
            t_all.append(t)
            break
env.close()
print(t_all)
print(sum(t_all)/len(t_all))