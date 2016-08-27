import sys
import csv
import sys
from time import time, strftime

import numpy as np
from scipy.sparse import *
from scipy.stats import randint as sp_randint

from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.calibration import CalibratedClassifierCV

from sklearn.preprocessing import Normalizer
from sklearn.linear_model import SGDClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC

OUTPUT_ALL = True

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

#clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
#calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)

candidates = [
  ("Ada300+nobigram+RS5", AdaBoostClassifier(n_estimators=300, random_state=5)),
  ("Ada400+nobigram+RS5", AdaBoostClassifier(n_estimators=400, random_state=5)),
]



ave_scores = []
k_fold = StratifiedKFold(y_train, n_folds=5, shuffle=True, random_state=17)
for name, clf in candidates:
  print 'Training %s' % name
  scores = [clf.fit(X_train[train], y_train[train]).score(X_train[cv].toarray(), y_train[cv]) for train, cv in k_fold]
  ave_scores.append(np.average(scores))
  print 'scores :', scores
  print 'average scores: %f' %  ave_scores[-1]
  
  if OUTPUT_ALL:
    try:
      preds = clf.fit(X_train, y_train).predict(X_test.toarray())
    except TypeError:
      preds = clf.fit(X_train.toarray(), y_train).predict(X_test.toarray())
    outcsv = name + '_' + strftime('%Y%m%d-%H%M%S') + '.csv'
    with open(outcsv, 'w') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(['Id', 'Prediction'])
      writer.writerows(enumerate(preds, 1))

if not OUTPUT_ALL:
  _, ind = max(enumerate(ave_scores), key=lambda x: x[1])
  preds = candidates[ind][1].fit(X_train, y_train).predict(X_test.toarray())
  outcsv = candidates[ind][0] + '_' + strftime('%Y%m%d-%H%M%S') + '.csv'
  with open(outcsv, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Prediction'])
    writer.writerows(enumerate(preds, 1))


  

