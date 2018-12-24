# coding: utf-8
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import *
from sklearn.tree import *
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import sys

if len(sys.argv) < 5:
    print('Not enough arguments.')
    exit(0)

mode = sys.argv[1]
train_path = sys.argv[2]
test_path = sys.argv[3]
output_path = sys.argv[4]

Train = pd.read_csv(train_path, header=None)
Test = pd.read_csv(test_path, header=None)

X_train = Train.drop(14, axis=1)
y_train = Train[14]
X_test = Test

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train[[1, 6, 13, 3, 5, 7, 8, 9]])

t1 = pd.DataFrame(enc.transform(X_train[[1, 6, 13, 3, 5, 7, 8, 9]]).toarray(), index=X_train.index) # encode categorical data
t2 = pd.DataFrame(enc.transform(X_test[[1, 6, 13, 3, 5, 7, 8, 9]]).toarray(), index=X_test.index) # encode categorical data
X_train_final = pd.concat([X_train.drop([1, 6, 13, 3, 5, 7, 8, 9], axis=1), t1], axis=1, ignore_index=True)
X_test_final = pd.concat([X_test.drop([1, 6, 13, 3, 5, 7, 8, 9], axis=1), t2], axis=1, ignore_index=True)

mmscaler = MinMaxScaler()
mmscaler.fit(X_train_final[[1]])
X_train_final[[1]] = mmscaler.transform(X_train_final[[1]])
X_test_final[[1]] = mmscaler.transform(X_test_final[[1]])

if mode == 'N':
    clf = GaussianNB(var_smoothing=0.05)
    clf.fit(X_train_final, y_train)
    pred = clf.predict(X_test_final)
    
elif mode == 'D':
    clf = DecisionTreeClassifier(max_depth=7, random_state=0)
    clf.fit(X_train_final, y_train)
    pred = clf.predict(X_test_final)
    
else:
    print('Undefined model.')
    exit(0)

with open(output_path, 'w') as fout:
    for i in pred:
        fout.write(str(int(i)) + '\n')

# N 0.8024385771910525 0.8016936534524475
# D 0.8607444077741108 0.8581797878280291

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier

# X = Train.drop(14, axis=1)
# y = Train[14]

# gnb_train = []
# gnb_test = []
# dt_train = []
# dt_test = []

# for i in range(0, 500):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=i)
    
# #     X_train = X_train.drop([1, 6, 13], axis=1) # 1, 6, 13 have missing value
# #     X_test = X_test.drop([1, 6, 13], axis=1)
    
#     enc = OneHotEncoder(handle_unknown='ignore')
#     enc.fit(X_train[[1, 6, 13, 3, 5, 7, 8, 9]])
# #     enc.fit(X_train[[3, 5, 7, 8, 9]])
    
#     t1 = pd.DataFrame(enc.transform(X_train[[1, 6, 13, 3, 5, 7, 8, 9]]).toarray(), index=X_train.index) # encode categorical data
#     t2 = pd.DataFrame(enc.transform(X_test[[1, 6, 13, 3, 5, 7, 8, 9]]).toarray(), index=X_test.index) # encode categorical data
#     X_train_final = pd.concat([X_train.drop([1, 6, 13, 3, 5, 7, 8, 9], axis=1), t1], axis=1, ignore_index=True)
#     X_test_final = pd.concat([X_test.drop([1, 6, 13, 3, 5, 7, 8, 9], axis=1), t2], axis=1, ignore_index=True)
    
# #     t1 = pd.DataFrame(enc.transform(X_train[[3, 5, 7, 8, 9]]).toarray(), index=X_train.index) # encode categorical data
# #     t2 = pd.DataFrame(enc.transform(X_test[[3, 5, 7, 8, 9]]).toarray(), index=X_test.index) # encode categorical data
# #     X_train_final = pd.concat([X_train.drop([3, 5, 7, 8, 9], axis=1), t1], axis=1, ignore_index=True)
# #     X_test_final = pd.concat([X_test.drop([3, 5, 7, 8, 9], axis=1), t2], axis=1, ignore_index=True)
    
#     mmscaler = MinMaxScaler()
#     mmscaler.fit(X_train_final[[1]])
#     X_train_final[1] = mmscaler.transform(X_train_final[[1]])
#     X_test_final[1] = mmscaler.transform(X_test_final[[1]])
        
# #     clf = GaussianNB(var_smoothing=0.05)
# #     clf.fit(X_train_final, y_train)
# # #     print(clf.score(X_train_final, y_train), clf.score(X_test_final, y_test))
    
# #     gnb_train.append(clf.score(X_train_final, y_train))
# #     gnb_test.append(clf.score(X_test_final, y_test))
    
# #     clf = DecisionTreeClassifier(random_state=0)
# #     clf.fit(X_train_final, y_train)
# # #     print(clf.score(X_train_final, y_train), clf.score(X_test_final, y_test))
    
# #     dt_train.append(clf.score(X_train_final, y_train))
# #     dt_test.append(clf.score(X_test_final, y_test))

#     clf = RandomForestClassifier(n_estimators=100)
#     clf.fit(X_train_final, y_train)
#     dt_train.append(clf.score(X_train_final, y_train))
#     dt_test.append(clf.score(X_test_final, y_test))