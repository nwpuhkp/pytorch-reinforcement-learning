import pandas as pd
from MFCC import *

class mfcc_env:
    df =None
    def __init__(self, csv_path='E:\pycharm\mfcc\state_v1.csv'):
        self.count = 0
        self.state = None
        self.action_space_dim = 7
        self.state_space_dim = 228
        if self.df is None:
            self.df = pd.read_csv(csv_path,sep=',',encoding='utf-8')
        self.job = self.df.loc[0]
        self.path = self.job[0]
        self.index = self.job[1]

    def state_loader(self,state_index):
        state = self.df.loc[state_index]
        mfcc = make_mfcc(state[0])
        index = state[1]
        return mfcc,index

    def step(self,action,index):
        self.count += 1
        # print(action)
        # print(index)
        reward = 1 if action % index == 0 else 0
        state,index = self.state_loader(self.count)
        done = True if self.count == 498 else False
        return state,reward,done,index

    def reset(self):
        self.__init__()
        self.state = make_mfcc(self.path)
        # print(self.index)

        return self.state,self.index