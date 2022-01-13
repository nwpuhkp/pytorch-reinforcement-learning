import cv2
import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


class QNetwork(nn.Module):
    def __init__(self,state_size,action_size,seed):
        super(QNetwork, self).__init__()
        self.seed = torch.cuda.manual_seed(seed)#在需要生成随机数据的实验中，每次实验都需要生成数据。设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。
        #具体网络可在以下更改(CNN)
        self.conv = nn.Sequential(
            #由于如果将一张图片作为状态输入信息，很多隐藏信息就会忽略（比如球往哪边飞），于是把连续的4帧图片作为状态输入
            nn.Conv2d(state_size[1],32,kernel_size=8,stride=4),#第一层卷积核8*8，stride=4，输出通道为32，ReLU
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2),#第二层卷积核4*4，stride=2，输出通道为64，ReLU
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1),#第三层卷积核3*3，stride=1，输出通道为64，ReLU
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(64*7*7,512),#第三层输出经过flat之后维度为3136，然后第四层连接一个512大小的全连接层
            nn.ReLU(),
            nn.Linear(512,action_size)#第五层为动作空间大小的输出层，Breakout游戏中为4，表示每种动作的概率
        )

    def forward(self,state):
        #构建一个由状态能映射到动作的网络
        conv_out = self.conv(state).view(state.size()[0],-1)
        return self.fc(conv_out)

def pre_process(observation):
    #压缩图片，构造输入
    x_t = cv2.cvtColor(cv2.resize(observation,(84,84)),cv2.COLOR_BGR2GRAY)#BGR转灰度图片
    ret,x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)#cv2.threshold (源图片, 阈值, 填充色, 阈值类型);阈值的二值化操作，大于阈值使用maxval表示，小于阈值使用0表示
    return np.reshape(x_t,(1,84,84)),x_t#reshape(a, newshape, order='C')

def stack_state(processed_obs):
    #四帧图片作一个状态
    return np.stack((processed_obs,processed_obs,processed_obs,processed_obs),axis=0)#重组四个array

if __name__ == '__main__':
    env = gym.make('Breakout-v0').unwrapped  # 建立环境
    state_size = env.observation_space.shape  # 通过gym模块输出Atari环境的游戏，状态空间都是（210, 160, 3），即210*160的图片大小，3个通道
    action_size = env.action_space.n

    print('形状：', state_size)
    print('操作数：', env.action_space.n)

    obs = env.reset()
    x_t,img = pre_process(obs)
    state = stack_state(img)
    print(np.shape(state[0]))

    plt.imshow(img, cmap='gray')
    #用cv2模块显示
    cv2.imshow('Breakout', img)
    cv2.waitKey(0)

    state = torch.randn(32,4,84,84)# (batch_size, color_channel, img_height,img_width)
    state_size = state.size()

    cnn_model = QNetwork(state_size,action_size=4,seed=1)
    outputs = cnn_model(state)
    print(outputs)

