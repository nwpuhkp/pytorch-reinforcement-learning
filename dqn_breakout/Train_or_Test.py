import time
from collections import deque

import cv2
import gym
import numpy as np
import torch
from matplotlib import pyplot as plt

from Agent import Agent
from ReplayMemory import ReplayMemory
from Qnet import stack_state

env = gym.make('Breakout-v0')
state_size = env.observation_space.shape
action_size = env.action_space.n
# print(state_size)
# print(action_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = Agent((32,4,84,84),action_size,seed=1)# state size (batch_size, 4 frames, img_height, img_width)
TRAIN = True  #训练还是测试


def pre_process(observation):#图片格式变换，灰度，二值化
    x_t = cv2.cvtColor(cv2.resize(observation, (84, 84)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    return x_t

def dqn(n_episodes=30000,max_t=40000,eps_start=1.0,eps_end=0.01,eps_decay=0.9995):
    #n_episodes:最大集数，max_t:每集最大步数最大帧数
    scores = [] #每集的分数
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(1,n_episodes+1):
        obs = env.reset()
        obs = pre_process(obs)
        state = stack_state(obs)

        score = 0
        for t in range(max_t):
            action = agent.act(state,eps)
            next_state, reward, done, _ = env.step(action)
            #后三帧和当前帧作为下一状态
            next_state = np.stack((state[1],state[2],state[3],pre_process(next_state)),axis=0)
            agent.step(state,action,reward,next_state,done)
            state = next_state
            score += reward

            if done:
                break

        scores_window.append(score)#保存最近的分数
        scores.append(score)

        eps = max(eps_end,eps_decay*eps) #减小eps
        print('\tEpsilon now : {:.2f}'.format(eps))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 1000 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print('\rEpisode {}\tThe length of replay buffer now: {}'.format(i_episode, len(agent.memory)))

        if np.mean(scores_window) >= 50.0:#达到50分代表合格
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,np.mean(scores_window)))

            torch.save(agent.qnet_local.state_dict(),'hkp_dqn_breakout_solved50points.pth')
            break

    torch.save(agent.qnet_local.state_dict(),'hkp_dqn_breakout_1.pth')
    return scores

if __name__ == '__main__':
    if TRAIN:
        start_time = time.time()
        scores = dqn().to(device)
        print('COST: {} min'.format((time.time() - start_time) / 60))
        print("Max score:", np.max(scores))

        #画图
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

    else:
        #加载训练模型
        agent.qnet_local.load_state_dict(torch.load('hkp_dqn_breakout_1.pth'))
        rewards = []
        for i in range(10):
            #玩10次
            total_reward = 0
            obs = env.reset()
            obs = pre_process(obs)
            state = stack_state(obs)

            for j in range(10000):#防卡帧
                action = agent.act(state)
                env.render()
                next_state,reward,done,_ = env.step(action)
                state = np.stack((state[1], state[2], state[3], pre_process(next_state)), axis=0)
                total_reward += reward

                time.sleep(0.01)
                if done:
                    rewards.append(total_reward)
                    break

        print("Test rewards are:", *rewards)
        print("Average reward:", np.mean(rewards))
        env.close()
#test




