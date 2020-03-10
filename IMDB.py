import sklearn.linear_model
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os
from pathlib import Path
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV


# Get the filenames from the training and testing datasets
train_pos = os.listdir(Path("aclImdb/train/pos"))
train_neg = os.listdir(Path("aclImdb/train/neg"))
test_pos = os.listdir(Path("aclImdb/test/pos"))
test_neg = os.listdir(Path("aclImdb/test/neg"))

# Create the training and test sets as lists of strings
train_data = []
for file in train_neg:
    path = os.path.join(Path("aclImdb/train/neg"), file)
    f = open(path, encoding="utf8")
    train_data.append(f.read())
    f.close()
    
for file in train_pos:
    path = os.path.join(Path("aclImdb/train/pos"), file)
    f = open(path, encoding="utf8")
    train_data.append(f.read())
    f.close()


test_data = []
for file in test_neg:
    path = os.path.join(Path("aclImdb/test/neg"), file)
    f = open(path, encoding="utf8")
    test_data.append(f.read())
    f.close()

for file in test_pos:
    path = os.path.join(Path("aclImdb/test/pos"), file)
    f = open(path, encoding="utf8")
    test_data.append(f.read())
    f.close()
    


# Create target lists with neg = 0 and pos = 1 (first half is neg, second half is pos)
train_target = [i//len(train_pos) for i in range(len(train_data))]
test_target = [i//len(test_pos) for i in range(len(test_data))]

# Shuffle positive and negative instances in both training and testing sets to randomize order
# The seed ensures that each pair of lists is shuffled the same way
random.seed(0)
random.shuffle(train_data)
random.seed(0)
random.shuffle(train_target)

random.seed(1)
random.shuffle(test_data)
random.seed(1)
random.shuffle(test_target)


# Make all lists numpy arrays
train_data = np.array(train_data)
train_target = np.array(train_target)
test_data = np.array(test_data)
test_target = np.array(test_target)


# Tokenize text by indexing each word in each document and associate it with its occurrence
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_data)

# Transform occurrence into tf-idf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


#print(train_data[5])
#print(test_data[2])
#print(X_train_tfidf.shape)

pipeline = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', LogisticRegression()) #step2 - classifier
])
clfs = []
clfs.append(LogisticRegression()) #c
clfs.append(DecisionTreeClassifier())
clfs.append(LinearSVC())#c
clfs.append(AdaBoostClassifier()) #learning rate
clfs.append(RandomForestClassifier())# n_estimators



for classifier in clfs:
    pipeline.set_params(clf = classifier)
    pipeline.fit(train_data, train_target)
    predicted = pipeline.predict(test_data)
    accuracy = np.mean(predicted == test_target)
    scores = cross_validate(pipeline,train_data, train_target)
    model_name = type(classifier).__name__
    print("-----------------------------------------------------------------")
    print(model_name)
    print(model_name+" accuracy: ", accuracy)
    for key, values in scores.items():
            print(model_name+" "+ key,' mean ', values.mean())
            print(model_name+" " + key,' std ', values.std())



#find best parameters
for classifier in clfs:
    pipeline.set_params(clf=classifier)
    if(isinstance(classifier, LogisticRegression) or isinstance(classifier, LinearSVC) ):
      cv_grid = GridSearchCV(pipeline, param_grid={
      'clf__C': np.linspace(0.1, 1.5, 25)
     })
    elif isinstance(classifier, AdaBoostClassifier):
      cv_grid = GridSearchCV(pipeline, param_grid={
      'clf__learning_rate': np.linspace(0.1, 1.5, 25)
     })
    elif isinstance(classifier, RandomForestClassifier):
      cv_grid = GridSearchCV(pipeline, param_grid={
      'clf__n_estimators': np.linspace(100, 300, 25)
     })
    else: pass
    cv_grid.fit(train_data, train_target)
    best_par = cv_grid.best_params_
    best_estimator = cv_grid.best_estimator_
    best_score = cv_grid.best_score_
    model_name = type(classifier).__name__
    print("-----------------------------------------------------------------")
    print(model_name)
    print(model_name + best_par)
    print(model_name + best_estimator)
    print(model_name + best_score)
    y_predict = cv_grid.predict(test_data)
    accuracy = accuracy_score(test_target, y_predict)
    print('Accuracy of the best' + model_name + 'after CV is %.6f%%' % (accuracy * 100))
