import sys
import csv
import sys
from time import time, strftime

import numpy as np
from scipy.sparse import *

from sklearn import metrics
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

def read_npz_data(filename):
  mats = np.load(filename)
  data = csr_matrix((mats['data'], mats['indices'], mats['indptr']), shape=(tuple(mats['shape'])))
  labels = mats['labels'] if 'labels' in mats else []
  return data, labels

def stacking(X_train, y_train, X_test):
  n_trees = 50
  n_folds = 5
  candidates = [
    RandomForestClassifier(n_estimators=n_trees, n_jobs=-1),
    ExtraTreesClassifier(n_estimators=n_trees, n_jobs=-1),
    GradientBoostingClassifier(n_estimators=n_trees),
    KNeighborsClassifier(),
    MultinomialNB(),
    AdaBoostClassifier(n_estimators=100),
    LogisticRegression(),
    SGDClassifier()
  ]
 
 
  k_fold = StratifiedKFold(y_train, n_folds=n_folds, shuffle=True, random_state=0)
  blend_train = np.zeros((X_train.shape[0], len(candidates))) 
  blend_test = np.zeros((X_test.shape[0], len(candidates)))
  for i, clf in enumerate(candidates):
    print 'Training classifier [%d]' % i 
    if n_folds > 0:
      for j, (train, cv) in enumerate(k_fold):
        print '[fold %d]' % j
        clf.fit(X_train[train], y_train[train])
        blend_train[cv, i] = clf.predict(X_train[cv])
        blend_test[:, i] = blend_test[:, i] + clf.predict(X_test)
      blend_test[:, i] = np.sign(blend_test[:, i]) + (blend_test[:, i] == 0)
    else:
      clf.fit(X_train, y_train)
      blend_train[:, i] = clf.predict(X_train)
      blend_test[:, i] = clf.predict(X_test)
   
    
  
  # stacking
  bclf = LogisticRegression()
  bclf.fit(blend_train, y_train)

  # predict
  bpreds = bclf.predict(blend_test)
  print 'Coefficient of features in the decision function'
  print bclf.coef_
  return bpreds 


# main program
if len(sys.argv) < 3:
  print '%s <train data>  <test data> ' % sys.argv[0]
  exit(1)


(X_train, y_train), (X_test, y_test) = map(read_npz_data, sys.argv[1:3])
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.toarray())
X_test = scaler.transform(X_test.toarray())

print 'Train set: %d X %d' % X_train.shape
print 'Test set: %d X %d' % X_test.shape


k_fold = StratifiedKFold(y_train, n_folds=5, shuffle=True, random_state=17)
scores = []
for i, (train, cv) in enumerate(k_fold):
  print 'Testing on fold [%d]' % i
  preds = stacking(X_train[train], y_train[train], X_train[cv])
  score = metrics.accuracy_score(y_train[cv], preds)
  print 'accuracy scores: %f' % score
  scores.append(score)

print 'Average score = %f' % np.average(np.array(scores))
 



