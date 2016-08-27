# Team BubblesandBlossom
# Machine Learning - Kaggle Competition


## Data set
This repo contains two sets of preprocessed data:
* fullbinary: one-hot representation of all features
  * fullbinary_train.npz (train data in sparse matrix)
  * fullbinary_test.npz  (test data in sparse matrix) 
  * fullbinary_features  (feature names)
* nobigram: no bigram features, one-hot representation 
  * nobigram_train.npz (train data in sparse matrix)
  * nobigram_test.npz  (test data in sparse matrix) 
  * nobigram_features  (feature names)

## Scripts
* feature_creation.py
* model_selection.py
* model_test.py
* param_tuning.py
* rf_tuning.py
* stacking.py
* final_predictions.py


## Run script: model_selection.py

How to tune parameters:
Model_selection.py contains a list called "candidates": 
candidates = [('RF1', RandomForestClassifier(n_estimators=900, random_state=1, n_jobs=-1))]

Here, 'RF1' is the name of this classifier; n_estimators, random_state and n_jobs are paremeters. The program will train all candidate classifiers in the list and compute their accuracies on a cross validation set.
 

###  Parameters to tune for RFC are:
* n_estimators:  the number of trees in the random forest (300, 900, 1200, 2000, 3000) 
* criterion: 'gini' or 'entropy'
* max_features: number of features to consider in split, ('auto', None, 'sqrt', 'log2')
* min_samples_leaf: minimum number of samples in newly created leaves (1, 5, 10, 15, 50) 
* random_state: seed of the random generator (must specify a number in order to reproduce the results!)
* n_jobs: the number of jobs/threads to run (-1 using all the cores).
* Other parameters to try: max_depth, min_samples_split 

### Run model_selection
    python model_selection.py fullbinary_train.npz fullbinary_test.npz 

or 

    python model_selection.py nobigram_train.npz nobigram_test.npz 

### Result of model_selection
* Output to stdin: each classifier's accuracy and training/testing time 
* Output to file: predictions for quiz of the best classifier 
