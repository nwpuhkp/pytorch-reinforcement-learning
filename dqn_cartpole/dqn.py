import math
import random
from collections import namedtuple, deque
from itertools import count

import gym

from torchvision.transforms import InterpolationMode

#载入gym的环境，用env.unwrapped可以得到原始的类，不会限制epoch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch import nn, optim

env = gym.make('CartPole-v0').unwrapped

#建立matplotlib
is_ipyhton = 'inline' in matplotlib.get_backend()
if is_ipyhton :
    from IPython import display

plt.ion()

#尽量调用gpu进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#单个转换的命名元组在环境中
Transition = namedtuple('Transition',('state','action','next_state','reward'))#Returns a new subclass of tuple with named fields.


class ReplayMemory(object):#有界的循环缓冲区，用来保存最近观察到的转变

    def __init__(self,capacity):#容量
        self.memory = deque([],maxlen=capacity)

    def push(self,*args):
        """保存一次中间交互"""
        self.memory.append(Transition(*args))

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)#随机选择批次转换进行训练

    def __len__(self):
        return len(self.memory)

#搭建神经网络
class DQN(nn.Module):
#接受当前和之前屏幕图像的差距，输出Q（s，左）和Q（s，右）。由为网络预测预期收益在收到输入情况下采取每个动作。
    def __init__(self,h,w,outputs):
        super(DQN,self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=5,stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16,32,kernel_size=5,stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,kernel_size=5,stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        #线性输入连接的数量取决于conv2d层的输出,因此输入图像的大小并计算它
        def conv2d_size_out(size,kernel_size = 5,stride = 2):
            return (size - kernel_size)//stride + 1


        convW = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))

        convH = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size = convW * convH * 32

        self.head = nn.Linear(linear_input_size,outputs)

    def forward(self,x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0),-1))


#输入提取
resize = T.Compose([T.ToPILImage(),#转换成PILImage
                    T.Resize(40,interpolation=InterpolationMode.BICUBIC), #缩小或放大需要把图片转换为高40的图片
                    T.ToTensor()])#格式转换成tensor，（H  W  C） in range（255）=>(C H W) in range(1.0)

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2 #世界总长度,有效世界的范围是：[-x_threshold, x_threshold]
    scale = screen_width / world_width #世界转屏幕系数：world_unit*scale = screen_unit
    return int(env.state[0] * scale + screen_width / 2.0)#世界中点在屏幕中间，所以偏移屏幕一半
#环境 env 的 state 返回有4个变量 (位置x，x加速度, 偏移角度theta, 角加速度)初值值是4个[-0.05, 0.05)的随机数

#gym要求的返回屏幕为 400x600x3，但有时更大例如 800x1200x3。将其转换为pytorch顺序 (CHW)。
def get_screen():
    screen = env.render(mode='rgb_array').transpose((2,0,1))
    #删减屏幕顶端部分
    _,screen_height,screen_width = screen.shape
    screen = screen[:,int(screen_height*0.4):int(screen_height*0.8)]
    view_width = int(screen_width*0.6)#宽度只截取60%，左右各截30%
    cart_location = get_cart_location(screen_width)

    if cart_location < view_width//2 :
        # 太靠左了，左边没有30%空间，则从最左侧截取
        slice_range = slice(view_width)

    elif cart_location > (screen_width - view_width // 2):
        #太靠右，从右边开始截取
        slice_range = slice(-view_width,None)

    else:
        # 左右两侧都有空间，则截小车在中间
        slice_range = slice(cart_location - view_width // 2,cart_location + view_width // 2)
    #去掉边缘，这样我们就有一个以购物车为中心的方形图像
    screen = screen[:,:,slice_range]
    #现在screen的格式是numpy数组，值范围[0, 255]，int8。而PIL接受的是float32的tensor，值范围[0.0, 1.0]，所以需要转换一下
    # screen = torch.from_numpy(np.float32(screen)/255)引起内存数据拷贝

    screen = np.ascontiguousarray(screen,dtype=np.float32) /255
    screen = torch.from_numpy(screen)
    #unsqueeze()的作用是在n维之前增加一个维度，这里是在0维之前增加一个维度
    return resize(screen).unsqueeze(0).to(device)#因为pytorch.nn.Conv2d() 的输入形式为(N(batch数), C（channel）, Y（高）, X（宽）)

env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('一次屏幕截图')
plt.show()

#训练(中间函数)
#初始化参数
#行动决策采用 epsilon greedy policy，就是有一定的比例，选择随机行为（否则按照网络预测的最佳行为行事）。这个比例从0.9逐渐降到0.05，按EXP曲线递减
BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 1 #概率从0.9开始
EPS_END = 0.01
EPS_DECAY = 0.001  #越小下降越快
TARGET_UPDATE = 10

init_screen = get_screen()
_,_,screen_height,screen_width = init_screen.shape

n_actions = env.action_space.n#从gym的action空间中获得操作数

policy_net = DQN(screen_height,screen_width,n_actions).to(device)
target_net = DQN(screen_height,screen_width,n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
#优化器
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    """
    选择动作
    select_action() 的作用就是 选择网络输出的2个值中的最大值（）或 随机数
    """
    global steps_done
    sample = random.random()#[0, 1)随机
    # epsilon greedy policy。EPS_END 加上额外部分，steps_done 越小，额外部分越接近0.9
    eps_threshold = EPS_END + (EPS_START-EPS_END)*math.exp(-1.*steps_done/EPS_DECAY)
    steps_done = steps_done + 1
    if sample > eps_threshold:
        with torch.no_grad():#选择使用网络来做决定。max返回 0:最大值和 1:索引
            #相当于做了一个CrossEntropy,谁大取谁的索引
            return  policy_net(state).max(1)[1].view(1,1)#return 0 if value[0] > value[1] else 1
            #pytorch 的 tensor.max() 返回所有维度的最大值及其索引，但如果指定了维度，就会返回namedtuple，包含各维度最大值及索引 (values=..., indices=...)
            #max(1)[1] 只取了索引值，也可以用 max(1).indices。view(1,1) 把数值做成[[1]] 的二维数组形式。为何返回一个二维 [[1]] ? 这是因为后面要把所有的state用torch.cat() 合成batch
    else:#选择一个随机数 0 或 1
        return torch.tensor([[random.randrange(n_actions)]],device=device,dtype=torch.long)

episode_durations = []   #维持时间长度

def plot_durations():#用于绘制情节持续时间的函数
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations,dtype=torch.float)
    plt.title('..........Training........')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 平均每100次迭代画出一幅图
    if len(durations_t)>=100:
        means = durations_t.unfold(0,100,1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99),means))#拼接数据
        plt.plot(means.numpy())

    plt.pause(0.001) #暂停来更新绘图

    if is_ipyhton:
        display.clear_output(wait=True)
        display.display(plt.gcf())


#训练函数
def optimize_model():
    '''
    大致过程：
    1.从memory列表里选取n个 （state, action, next_state, reward）
    2.用net获取state的Y[0, 1]（net输出为2个值），再用action选出结果y
    3.用net获取next_state获取Y'[0,1]，取最大值 y'。如果state没有对应的next_state，则 y' = 0
    4.用公式算出期望y：\hat y = \gamma y' + reward （常量 \gamma = 0.9）
    5.用smoothl1loss计算误差
    6.用RMSprop 反向传导优化网络
    期望y的计算方法就是把next_state的net结果，直接乘一个0.9然后加上奖励。
    如果有 next_state，就是1，如果next_state为None，奖励是0。
    '''
    if len(memory)<BATCH_SIZE:#记忆池满才开始训练
        return
    #转化为batch
    transitions = memory.sample(BATCH_SIZE)#抽样
    batch = Transition(*zip(*transitions))#转换成一批次

    non_final_mask = torch.tensor(tuple(map(lambda s:s is not None,
                                        batch.next_state)),device=device,dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                      if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    #计算Q(s_t, a)，选择动作
    state_action_values = policy_net(state_batch).gather(1,action_batch)

    # 计算下一步的所有动作的价值V（s_{t+1}）
    next_state_values = torch.zeros(BATCH_SIZE,device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # 计算预期的Q值
    expected_state_action_values = (next_state_values*GAMMA)+reward_batch

    # 计算Huber loss，损失函数采用smoothllloss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,expected_state_action_values.unsqueeze(1))

    # 优化模型
    optimizer.zero_grad()#清理所有梯度
    loss.backward()#反向传播
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1,1)  #将所有的梯度限制在-1，1之间
    optimizer.step()#更新模型的参数

#开始训练
num_episodes = 800 #迭代次数
for i_episodes in range(num_episodes):
    #初始化环境和状态
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen-last_screen#将状态定义为当前状态和上一步的状态差
    for t in count():
        #选择并执行一个动作
        action = select_action(state)
        _,reward,done,_ = env.step(action.item()) #从环境中获取奖励
        reward = torch.tensor([reward],device=device) #将奖励转换为tensor

        #观察新的状态，确定下个状态
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # 将转换保存起来
        memory.push(state,action,next_state,reward)

        # 切换到下一状态
        state = next_state

        # 优化模型
        optimize_model()
        # 一次游戏结束，就画图显示
        if done:
            episode_durations.append(t+1)
            plot_durations()
            break
        #更新目标网络，复制DQN中的所有权重和偏差
    if i_episodes % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('完成')
env.render()
env.close()
plt.ioff()
plt.show()









