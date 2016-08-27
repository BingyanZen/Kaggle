import sys
import csv
from time import time, strftime

import numpy as np
from scipy.sparse import *
from scipy.stats import randint as sp_randint

from operator import itemgetter
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV


def read_npz_data(filename):
  mats = np.load(filename)
  data = csr_matrix((mats['data'], mats['indices'], mats['indptr']), shape=(tuple(mats['shape'])))
  labels = mats['labels']
  return data, labels
  

# main program
if len(sys.argv) < 3:
  print '%s <train data>  <test data> ' % sys.argv[0]
  exit(1)

(X_train, y_train), (X_test, y_test) = map(read_npz_data, sys.argv[1:])
print 'Number of training samples: ', X_train.shape[0]
print 'Number of features: ', X_train.shape[1] - 1

# Utility function to report best scores
def report(grid_scores, n_top=3):
  top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
  for i, score in enumerate(top_scores):
    print("Model with rank: {0}".format(i + 1))
    print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
          score.mean_validation_score,
          np.std(score.cv_validation_scores)))
    print("Parameters: {0}".format(score.parameters))
    print("")


clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=1, random_state=17)

param_dist = {"max_features": [None, "auto", "sqrt"],
              "min_samples_leaf": [1, 10, 50],
              "criterion": ["gini", "entropy"]}    
n_iter_search = 10
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)

start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)

best_clf = random_search.best_estimator_
test_pred = best_clf.predict(X_test.toarray())

outcsv = 'RandomCV_' + strftime('%Y%m%d-%H%M%S') + '.csv'
with open(outcsv, 'w') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(['Id', 'Prediction'])
  writer.writerows(enumerate(test_pred, 1))
  

