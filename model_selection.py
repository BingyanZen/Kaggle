import sys
import csv
import sys
from operator import itemgetter
from time import time, strftime

import numpy as np
from scipy.sparse import *
from scipy.stats import randint as sp_randint

from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier


from sklearn.preprocessing import Normalizer
from sklearn.linear_model import SGDClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC

VOTE = False
PRED = True
RANKING = False 


# main program
if len(sys.argv) < 3:
  print '%s <train data>  <test data> ' % sys.argv[0]
  exit(1)


def read_npz_data(filename):
  mats = np.load(filename)
  data = csr_matrix((mats['data'], mats['indices'], mats['indptr']), shape=(tuple(mats['shape'])))
  labels = mats['labels'] if 'labels' in mats else []
  return data, labels

X_train, y_train = read_npz_data(sys.argv[1])
# normalize input vectors
norm = Normalizer(norm='l2')
X_train = norm.fit_transform(X_train)


print 'Number of training samples: ', X_train.shape[0]
print 'Number of features: ', X_train.shape[1]


X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33, random_state=32)



def benchmark(clf):
  clf_descr = str(clf)
  print '=' * 80
  print 'Training %s: ' % clf_descr
  start_t = time()
  clf.fit(X_train, y_train)
  print 'train time: %0.3fs' % (time() - start_t)
  start_t = time()
  try:
    pred = clf.predict(X_cv)
  except TypeError:
    pred = clf.predict(X_cv.toarray())
  print 'test time: %0.3fs' % (time() - start_t)
  score = metrics.accuracy_score(y_cv, pred)
  print 'accuracy: %f' % score
  return score, pred

# candidate models
candidates = [
    ("RF+5RS+auto+gini+minLS1+max5", RandomForestClassifier(n_estimators=900, n_jobs=1, random_state=5,criterion= 'gini',min_samples_leaf=1,max_features=5)),
    ("RF+5RS+auto+gini+minLS1+max10", RandomForestClassifier(n_estimators=900, n_jobs=1, random_state=5,criterion= 'gini',min_samples_leaf=1,max_features=10)),    
    ("RF+5RS+auto+gini+minLS1++max20", RandomForestClassifier(n_estimators=900, n_jobs=1, random_state=5,criterion= 'gini',min_samples_leaf=1,max_features=20)),
    ("RF+5RS+auto+gini+minLS1++max25", RandomForestClassifier(n_estimators=900, n_jobs=1, random_state=5,criterion= 'gini',min_samples_leaf=1,max_features=25)),
    ("RF+5RS+auto+gini+minLS1+max30", RandomForestClassifier(n_estimators=900, n_jobs=1, random_state=5,criterion= 'gini',min_samples_leaf=1,max_features=30))
]


# parameters of RandomForestClassifier to try are
# n_estimators, criterion, max_features, min_samples_leaf
# Must specify a random_state!

results = []
for name, clf in candidates:
  results.append(benchmark(clf))
best_clf, __ = max(enumerate(results), key=lambda x: x[1][0])


# ensemble all models
if VOTE:
  vote_pred = np.zeros(X_cv.shape[0])
  weight = 3
  for i, (score, pred) in enumerate(results):
    vote_pred = vote_pred + (weight if i == best_clf else 1) * pred
  vote_pred = np.sign(vote_pred) - (vote_pred == 0)

  print '=' * 80
  print 'Voting Ensemble (%d * Best vs. Rest)' % weight
  print '  Best clf is %s' % candidates[best_clf][0]
  score = metrics.accuracy_score(y_cv, vote_pred)
  print 'accuracy: %f' % score
 
del X_train, y_train, X_cv, y_cv 


# predict X_test with best clf 
if PRED: 
  clf = candidates[best_clf][1]
  clf_name = candidates[best_clf][0]
  print 'Training %s with full data' % clf_name
  (X_train, y_train), (X_test, y_test) = map(read_npz_data, sys.argv[1:3])

  # normalize input vectors
  norm = Normalizer(norm='l2')
  X_train = norm.fit_transform(X_train)
  X_test = norm.transform(X_test)


  clf.fit(X_train, y_train)
  try:
    test_pred = clf.predict(X_test)
  except TypeError:
    test_pred = clf.predict(X_test.toarray())

  outcsv = clf_name + '_' + strftime('%Y%m%d-%H%M%S') + '.csv'
  with open(outcsv, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Prediction'])
    writer.writerows(enumerate(test_pred, 1))

  

############################################OUTDATED##############################################
"""  
if RANKING:
  clf = candidates[best_clf][1]
  clf_name = candidates[best_clf][0]

  print 'Training %s with full data' % clf_name
  (X_train, y_train), (X_test, y_test) = map(read_npz_data, sys.argv[1:3])

  clf.fit(X_train, y_train)
  importances = clf.feature_importances_

  print 'Output ranking'
  with open('fullbinary_features.txt', 'r') as infile:
    features = infile.read().split(',')

  with open('fullbinary_ranking.txt', 'w') as outfile:
    for ind, score in sorted(enumerate(importances), key=lambda x: x[1]):
      outfile.write('%-10.2f %s\n' % (score, features[ind]))
    

    
     
  

    ("GB", GradientBoostingClassifier(n_estimators=1000, random_state=2)),
    ("ET1", ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, random_state=3)),
    ("ET2", ExtraTreesClassifier(n_estimators=1000, criterion="entropy", n_jobs=-1, random_state=4))]

for penalty in ['l2', 'l1']:
  print ''
  print '%s penalty' % penalty
  results.append(benchmark(SGDClassifier(penalty=penalty)))
  results.append(benchmark(LogisticRegression(penalty=penalty)))
"""
 
