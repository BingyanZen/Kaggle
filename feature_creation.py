
import numpy as np
import pandas as pd
import sys

from sklearn.feature_extraction import DictVectorizer

#filter_cols = ['label', '18', '20', '23', '25', '26', '58']
filter_cols = ['label']

if len(sys.argv) < 4:
  print '%s <train data> <test data> <dataset name>' % sys.argv[0]
  exit(1)

# read in training/test data
(df_train, df_test) = map(pd.read_csv, sys.argv[1:3])
feature_cols = [col for col in df_train.columns if col not in filter_cols] 

# clean up data   
df_train['7'] = df_train['7'].map({'f': 0.0, 'g': 1.0})
df_train['16'] = df_train['16'].map({'f': 0.0, 'g': 1.0})
df_test['7'] = df_test['7'].map({'f': 0.0, 'g': 1.0})
df_test['16'] = df_test['16'].map({'f': 0.0, 'g': 1.0})

# some of numerical features only 0 or 1?
 
X_train = df_train[feature_cols].to_dict(orient='records')
y_train = df_train['label']
X_test = df_test[feature_cols].to_dict(orient='records')
del df_train
del df_test

# transform to one-hot
v = DictVectorizer(sparse=True)
X_train = v.fit_transform(X_train)
X_test = v.transform(X_test)
feature_names = v.get_feature_names()

# write data to npz, feature names to txt
def save_sparse_csr(filename):
  np.savez(filename + '_train', data=X_train.data, indices=X_train.indices, indptr=X_train.indptr, shape=X_train.shape, labels=y_train)
  np.savez(filename + '_test', data=X_test.data, indices=X_test.indices, indptr=X_test.indptr, shape=X_test.shape)


print "Writing dataset  %s" % sys.argv[3] 
print "  After one-hot transformation: %d X %d " % X_train.shape
save_sparse_csr(sys.argv[3])
with open(sys.argv[3] + '_features.txt', 'w') as outfile:
  outfile.write(','.join(feature_names))






