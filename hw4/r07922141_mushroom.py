# coding: utf-8
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import *
from sklearn.tree import *
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

X_train = Train.drop(22, axis=1)
y_train = Train[22]
X_test = Test

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(X_train)
X_train_one_hot = enc.transform(X_train)

if mode == 'N':
    clf = MultinomialNB(alpha=1.0e-5)
    clf.fit(X_train_one_hot.toarray(), y_train)
    pred = clf.predict(enc.transform(X_test).toarray())
    
elif mode == 'D':
    clf = DecisionTreeClassifier()
    clf.fit(X_train_one_hot.toarray(), y_train)
    pred = clf.predict(enc.transform(X_test).toarray())
    
else:
    print('Undefined model.')
    exit(0)

with open(output_path, 'w') as fout:
    for i in pred:
        fout.write(str(int(i)) + '\n')


# # Gaussian NB (0.9909230769230769 0.9919950738916257)
# clf = GaussianNB(var_smoothing=1.0e-2)
# # clf = GaussianNB()
# clf.fit(X_train_one_hot.toarray(), y_train)
# train_score = clf.score(X_train_one_hot.toarray(), y_train)
# test_score = clf.score(enc.transform(X_test).toarray(), y_test)
# print(train_score, test_score)
# # 0.956 0.9550492610837439


# # Multinomial NB (0.9964615384615385 0.9963054187192119)
# clf = MultinomialNB(alpha=1.0e-5)
# # clf = MultinomialNB()
# clf.fit(X_train_one_hot.toarray(), y_train)
# train_score = clf.score(X_train_one_hot.toarray(), y_train)
# test_score = clf.score(enc.transform(X_test).toarray(), y_test)
# print(train_score, test_score)
# # 0.9504615384615385 0.9562807881773399


# # Bernuolli NB (0.9946153846153846 0.9932266009852216)
# # clf = BernoulliNB(alpha=1.0e-8)
# clf = BernoulliNB()
# clf.fit(X_train_one_hot.toarray(), y_train)
# train_score = clf.score(X_train_one_hot.toarray(), y_train)
# test_score = clf.score(enc.transform(X_test).toarray(), y_test)
# print(train_score, test_score)
# # 0.9376923076923077 0.9445812807881774


# # Complement NB (0.9964615384615385 0.9956896551724138)
# clf = ComplementNB(alpha=1.0e-5)
# # clf = ComplementNB()
# clf.fit(X_train_one_hot.toarray(), y_train)
# train_score = clf.score(X_train_one_hot.toarray(), y_train)
# test_score = clf.score(enc.transform(X_test).toarray(), y_test)
# print(train_score, test_score)
# # 0.9515384615384616 0.9550492610837439


# # Decision Tree (1.0 1.0)
# clf = DecisionTreeClassifier()
# clf.fit(X_train_one_hot.toarray(), y_train)
# train_score = clf.score(X_train_one_hot.toarray(), y_train)
# test_score = clf.score(enc.transform(X_test).toarray(), y_test)
# print(train_score, test_score)

# from sklearn import tree
# tree.export_graphviz(clf, out_file='mushroom_tree.dot') 