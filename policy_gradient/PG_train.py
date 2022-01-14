import argparse
from itertools import count

import gym
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical


# ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
# name or flags - 选项字符串的名字或者列表，例如 foo 或者 -f, --foo。
# action - 命令行遇到参数时的动作，默认值是 store。
# default - 不指定参数时的默认值。
# type - 命令行参数应该被转换成的类型。
# help - 参数的帮助信息，当指定为 argparse.SUPPRESS 时表示不显示该参数的帮助信息.
# metavar - 在 usage 说明中的参数名称，对于必选参数默认就是参数名称，对于可选参数默认是全大写的参数名称.

parser = argparse.ArgumentParser(description='PyTorch PG example')#创建 ArgumentParser() 对象

parser.add_argument('--gamma',type=float,default=0.99,metavar='G',help='discount factor(default:0.99)')#调用 add_argument() 方法添加参数
parser.add_argument('--seed',type=int,default=543,metavar='N',help='random seed (default:543)')
parser.add_argument('--log-interval',type=int,default=10,metavar='N',help='interval between training status logs(default:10)')
parser.add_argument('--render',action='store_true',help='render the environment')

args = parser.parse_args()#使用 parse_args() 解析添加的参数

env = gym.make('CartPole-v1')#创建模型
#在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。如果设置初始化，则每次初始化都是固定的。
env.seed(args.seed)#如果不给相同的种子，它将产生不同的随机数序列。由于机器学习受经验驱动，因此可重复性非常重要。
torch.manual_seed(args.seed) #为CPU设置种子用于生成随机数，以使得结果是确定的

class Policy(nn.Module):
    def __init__(self):
        super(Policy,self).__init__()
        self.affine1 = nn.Linear(4,128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128,2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self,x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        #softmax,输出概率，分值大的经常取到，分值小的也偶尔能够取到。
        return F.softmax(action_scores,dim=1)

policy = Policy()
optimizer = optim.Adam(policy.parameters(),lr = 0.01)
eps = np.finfo(np.float64).eps.item()


def select_action(state):
    #把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    #如果 probs 是长度为 K 的一维列表，则每个元素是对该索引处的类进行抽样的相对概率。如果 probs 是二维的，它被视为一批概率向量。
    m = Categorical(probs)#按照传入的probs中给定的概率，在相应的位置处进行取样，取样返回的是该位置的整数索引。\
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    #item(),显示精度上的区别？更精确
    return action.item()#取出单元素张量的元素值并返回该值，保持原元素类型不变。,即：原张量元素为整形，则返回整形，原张量元素为浮点型则返回浮点型，


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:#逆序
        R = r + args.gamma * R
        returns.insert(0,R)#list.insert(index, obj)index:对象 obj 需要插入的索引位置。obj:要插入列表中的对象。
    returns = torch.tensor(returns)
    # mean()函数功能：求取均值,std()函数返回请求轴上的样品标准偏差。
    returns = (returns - returns.mean())/(returns.std() + eps)
    for log_prob,R in zip(policy.saved_log_probs,returns):#zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]#del删除的是变量，而不是数据。[:]全选
    del policy.saved_log_probs[:]

def main():
    running_reward = 10
    for i_epsode in count(1):
        state,ep_reward = env.reset(),0
        for t in range(1,10000):
            action = select_action(state)
            state,reward,done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_epsode % args.log_interval == 0:
            print('Episode {}\tlast reward :{:.2f}\tAverage reward: {:.2f}'.format(i_epsode,ep_reward,running_reward))

        if running_reward > env.spec.reward_threshold:#成功阈值500
        #if running_reward > 100:  # 试验训练到100步

            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))

            torch.save(policy.state_dict(),'hkp_PG_train4.pth')
            break

if __name__ == '__main__':
    main()



