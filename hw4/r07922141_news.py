# coding: utf-8
import pandas as pd
import numpy as np
import sklearn.naive_bayes as nb
from sklearn import tree
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

# Training data
X_train = Train.drop(23909, axis=1)
y_train = Train[23909]
X_test = Test

if mode == 'N':
    clf = nb.MultinomialNB(alpha=0.05)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    pred = pred.astype(int)
    
elif mode == 'D':
    clf = tree.DecisionTreeClassifier(max_depth=55, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    pred = pred.astype(int)

else:
    print('Undefined model.')
    exit(0)

with open(output_path, 'w') as fout:
    for i in pred:
        fout.write(str(int(i)) + '\n')


# # Gaussian NB (0.9702187063750581 0.8097902097902098)
# clf = nb.GaussianNB()
# clf.fit(X_train, y_train)
# train_score = clf.score(X_train, y_train)
# test_score = clf.score(X_test, y_test)
# print(train_score, test_score)


# # Multinomial NB (0.9744067007910656 0.8944055944055944)
# clf = nb.MultinomialNB(alpha=0.05)
# # clf = nb.MultinomialNB()
# clf.fit(X_train, y_train)
# train_score = clf.score(X_train, y_train)
# test_score = clf.score(X_test, y_test)
# print(train_score, test_score)


# # Bernuolli NB (0.8529548627268497 0.7678321678321678)
# clf = nb.BernoulliNB()
# clf.fit(X_train, y_train)
# train_score = clf.score(X_train, y_train)
# test_score = clf.score(X_test, y_test)
# print(train_score, test_score)


# # Complement NB (0.9725453699395068 0.8839160839160839)
# clf = nb.ComplementNB(alpha=0.035)
# # clf = nb.ComplementNB()
# clf.fit(X_train, y_train)
# train_score = clf.score(X_train, y_train)
# test_score = clf.score(X_test, y_test)
# print(train_score, test_score)


# # Gridsearch (Multinomial NB get best score)
# from sklearn.model_selection import GridSearchCV
# import numpy as np

# alpha = np.linspace(0.0, 1.0, 100)
# mx = 0
# for n in [True, False]:
#     for i in alpha:
#         clf = nb.ComplementNB(norm=n, alpha=i)
#         clf.fit(X_train, y_train)
#         score = clf.score(X_test, y_test)       
#         if score > mx:
#             print(n, i, score)
#             mx = score
            
# mx = 0
# for i in alpha:
#     clf = nb.MultinomialNB(alpha=i)
#     clf.fit(X_train, y_train)
#     score = clf.score(X_test, y_test)
#     if score > mx:
#         print(i, score)
#         mx = score


# # Decision Tree (0.9711493718008376 0.6286713286713287)
# clf = tree.DecisionTreeClassifier(max_depth=55, random_state=42)
# clf.fit(X_train, y_train)
# train_score = clf.score(X_train, y_train)
# test_score = clf.score(X_test, y_test)
# print(train_score, test_score)


# # Gridsearch in Decision Tree
# max_vi = 0
# max_v = 0
# for i in range(1, 150):
#     clf = tree.DecisionTreeClassifier(max_depth=i, random_state=42)
#     clf.fit(X_train, y_train)
#     train_score = clf.score(X_train, y_train)
#     test_score = clf.score(X_test, y_test)
#     if test_score > max_vi and train_score > max_v:
#         print(i, train_score, test_score)
#         max_vi = test_score
#         max_v = train_score