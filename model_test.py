import sys
import csv
import sys
from time import time, strftime

import numpy as np
from scipy.sparse import *
from scipy.stats import randint as sp_randint

from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier


from sklearn.preprocessing import Normalizer
from sklearn.linear_model import SGDClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC


# main program
if len(sys.argv) < 3:
  print '%s <train data>  <test data> ' % sys.argv[0]
  exit(1)


def read_npz_data(filename):
  mats = np.load(filename)
  data = csr_matrix((mats['data'], mats['indices'], mats['indptr']), shape=(tuple(mats['shape'])))
  labels = mats['labels'] if 'labels' in mats else []
  return data, labels

(X_train, y_train), (X_test, y_test) = map(read_npz_data, sys.argv[1:3])

print 'Train set: %d X %d' % X_train.shape
print 'Test set: %d X %d' % X_test.shape

candidates = [
    ("RF200+nobigram+RS10", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=10)),
    ("RF200+entropy+nobigram+RS10", RandomForestClassifier(n_estimators=200, criterion='entropy', n_jobs=-1, random_state=10)),
]


#norm = Normalizer(norm='l2')
#X_train = norm.fit_transform(X_train)
#X_test = norm.transform(X_test)

k_fold = KFold(n=X_train.shape[0], n_folds=4, shuffle=True, random_state=17)
for name, clf in candidates:
  print 'Training %s' % name
  scores = [clf.fit(X_train[train], y_train[train]).score(X_train[cv], y_train[cv]) for train, cv in k_fold]
  print 'scores :', scores
  print 'average scores: %f' %  np.average(scores)
  preds = clf.fit(X_train, y_train).predict(X_test.toarray())
  outcsv = name + '_' + strftime('%Y%m%d-%H%M%S') + '.csv'
  with open(outcsv, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Prediction'])
    writer.writerows(enumerate(preds, 1))

