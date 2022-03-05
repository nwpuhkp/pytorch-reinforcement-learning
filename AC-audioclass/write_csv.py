import csv
from collections import namedtuple

import numpy as np

f = open('E:\pycharm\mfcc\state_v1.csv','w',newline='')
csv_writer = csv.writer(f)

for i in range(1000):
    t = np.random.randint(0,6)
    # print(t)
    buffer = [['./data/A.wav',0],
              ['./data/B.wav',1],
              ['./data/C.wav',2],
              ['./data/D.wav',3],
              ['./data/E.wav',4],
              ['./data/F.wav',5]]
    path = buffer[t][0]
    index = buffer[t][1]
    S = namedtuple('S',['path','index'])
    s = S(path = path,index= index)
    csv_writer.writerow([s.path,s.index])

f.close()