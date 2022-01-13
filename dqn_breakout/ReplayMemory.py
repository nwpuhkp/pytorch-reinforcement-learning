import random
from collections import deque, namedtuple

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ReplayMemory:
    #存储多个样例随机抽取
    def __init__(self,action_size,memory_size,batch_size,seed):#每个动作的维度，存储空间大小，每个训练批次的大小，随机种子
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)#当限制长度的deque增加超过限制数的项时，另一边的项会自动删除
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",field_names=["state","action","reward","next_state","done"])#不能为元组内部的数据进行命名，所以并不知道一个元组所要表达的意义，所以引入了collections.namedtuple，来构造一个带字段名的元组。
        self.seed = random.seed(seed)


    def add(self,state,action,reward,next_state,done):
        #把一个新的experience添加到memory中来
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)

    def sample(self):
        #从memory中随机抽样
        experiences = random.sample(self.memory,k=self.batch_size)
        #torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)#vstack竖直拼接，hstack水平拼接
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(device)

        return (states,actions,rewards,next_states,dones)

    def __len__(self):
        #返回memory当前大小
        return len(self.memory)


# print(device)