import pandas as pd
from MFCC import *

csv_path = 'state_v1.csv'
state = pd.read_csv(csv_path,sep=',',encoding='utf-8')
state = state.loc[0]
print(state.index)
mfcc = make_mfcc(state[0])
print(mfcc)
k = mfcc[:19]
print(k.shape)
# print(mfcc[:19][12])