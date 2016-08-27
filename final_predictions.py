#!/usr/bin/env python
from __future__ import print_function
import csv
import numpy as np
import pandas as pd
from sys import argv, exit, stderr
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer

if len(argv) < 4:
  print('Usage: python %s DATAFILE QUIZFILE OUTPUTFILE' % argv[0], file=stderr)
  exit(1)

# read in training/test data frames
filter_cols = ['label', '18', '20', '23', '25', '26', '58', '35']  # filter out bigram features
(df_train, df_test) = map(pd.read_csv, argv[1:3])
feature_cols = [col for col in df_train.columns if col not in filter_cols] 

# clean up data frames 
df_train['7'] = df_train['7'].map({'f': 0.0, 'g': 1.0})
df_train['16'] = df_train['16'].map({'f': 0.0, 'g': 1.0})
df_test['7'] = df_test['7'].map({'f': 0.0, 'g': 1.0})
df_test['16'] = df_test['16'].map({'f': 0.0, 'g': 1.0})

 
X_train = df_train[feature_cols].to_dict(orient='records')
y_train = df_train['label']
X_test = df_test[feature_cols].to_dict(orient='records')
del df_train
del df_test

# transform to one-hot
v = DictVectorizer(sparse=True)
X_train = v.fit_transform(X_train)
X_test = v.transform(X_test)
print('Train set: %d X %d' % X_train.shape)
print('Test set: %d X %d' % X_test.shape)


# train and predict
clf = RandomForestClassifier(n_estimators=900, n_jobs=-1, random_state=5)    
preds = clf.fit(X_train, y_train).predict(X_test.toarray())

# write final predictions 
with open(argv[-1], 'w') as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(['Id', 'Prediction'])
  writer.writerows(enumerate(preds, 1))
