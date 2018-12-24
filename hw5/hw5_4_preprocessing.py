#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import sys

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

df_train = pd.read_csv(sys.argv[1], header=None)
df_test = pd.read_csv(sys.argv[2], header=None)
y_train = df_train[14]
df_train = df_train.drop([14], axis=1)


enc_idx = [c for c in df_train.columns if c not in df_train.describe().keys()]
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df_train[enc_idx])

t1 = pd.DataFrame(enc.transform(df_train[enc_idx]).toarray(), index=df_train.index) # encode categorical data
t2 = pd.DataFrame(enc.transform(df_test[enc_idx]).toarray(), index=df_test.index) # encode categorical data
df_train_final = pd.concat([df_train.drop(enc_idx, axis=1), t1], axis=1, ignore_index=True)
df_test_final = pd.concat([df_test.drop(enc_idx, axis=1), t2], axis=1, ignore_index=True)

scaler = MinMaxScaler()
df_train_final[[0,1,2,3,4,5]] = scaler.fit_transform(df_train_final[[0,1,2,3,4,5]])
df_test_final[[0,1,2,3,4,5]] = scaler.transform(df_test_final[[0,1,2,3,4,5]])

with open(sys.argv[1]+'.tr', 'w') as fout:
    for i, j in zip(df_train_final.iterrows(), y_train):
        label = j
        string = str(j) + ' '
        for idx, k in enumerate(i[1]):
            if k > 0 and idx != 107:
                string += (str(idx+1)+':'+str(k)+' ')
            elif k > 0 and idx == 107:
                string += (str(idx+1)+':'+str(k))
        fout.write(string+'\n')

with open(sys.argv[2]+'.te', 'w') as fout:
    for i in df_test_final.iterrows():
        string = '0 '
        for idx, k in enumerate(i[1]):
            if k > 0 and idx != 107:
                string += (str(idx+1)+':'+str(k)+' ')
            elif k > 0 and idx == 107:
                string += (str(idx+1)+':'+str(k))
        fout.write(string+'\n')


# In[46]:


# from sklearn.svm import SVC

# clf = SVC(gamma='auto', kernel='linear')
# clf.fit(df_train_final, y_train)


# In[9]:


# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV


# In[42]:


# params_grid = {'C': [i for i in range(210, 230, 2)],
# #               'gamma': [0.0001, 0.001, 0.01, 0.1],
#               'kernel':['linear','rbf'] }

# grid_clf = GridSearchCV(SVC(gamma='scale'), params_grid, verbose=30, iid=True, n_jobs=12, cv=3)
# grid_clf.fit(df_train_final, y_train)


# In[45]:


# from sklearn.model_selection import cross_val_score
# clf = SVC(kernel='rbf', C=100, gamma=0.01)
# scores = cross_val_score(clf, df_train_final, y_train, cv=5)
# scores
# clf = SVC(kernel='rbf', C=200, gamma='auto')
# scores = cross_val_score(clf, df_train_final, y_train, cv=5)
# scores
# clf = SVC(kernel='linear', C=212, gamma='scale')
# scores = cross_val_score(clf, df_train_final, y_train, cv=5)
# scores


# In[ ]: