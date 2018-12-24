#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import string
import sys

df_train = pd.read_csv(sys.argv[1], header=None)
df_test = pd.read_csv(sys.argv[2], header=None)

df_train[0] = df_train[0].apply(lambda x: 0 if x == 'I' else 1 if x == 'F' else 2)
df_test[0] = df_test[0].apply(lambda x: 0 if x == 'I' else 1 if x == 'F' else 2)

def Formatting(df):
    lst = []

    for i in df.iterrows():
        label = int(i[1][8])
        string = str(label) + ' '
        for j in range(1, 8):
            if j == 7:
                string += (str(j)+':'+str(i[1][j])+' '+str(int(i[1][0]+8))+':1.0')
            else:
                string += (str(j)+':'+str(i[1][j])+' ')
        lst.append(string)
    return lst

test = Formatting(df_test)
train = Formatting(df_train)

with open(sys.argv[1]+'.tr', 'w') as fout:
    for i in train:
        fout.write(i+'\n')
        
with open(sys.argv[2]+'.te', 'w') as fout:
    for i in test:
        fout.write(i+'\n')