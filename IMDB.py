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

print("Analyzing IMDB dataset .......................................................")
#fit data with different models using pipelines and forloop

pipeline = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', LogisticRegression()) #step2 - classifier
])
clfs = []



clfs.append(DecisionTreeClassifier())
clfs.append(LinearSVC())#c
clfs.append(AdaBoostClassifier()) #learning rate
clfs.append(RandomForestClassifier())# n_estimators
clfs.append(LogisticRegression()) #c


best_model = ""
best_acc = 0
best_score = 0
print("Test for different models:................................................")
for classifier in clfs:
    pipeline.set_params(clf = classifier)
    pipeline.fit(train_data, train_target)
    predicted = pipeline.predict(test_data)
    accuracy = np.mean(predicted == test_target)
    scores = cross_validate(pipeline,train_data, train_target)
    model_name = type(classifier).__name__
    if(accuracy > best_acc):
        best_acc = accuracy
        best_score = np.mean(scores['test_score'])
        best_model = model_name
    print("-----------------------------------------------------------------")
    print(model_name)
    print(model_name+" accuracy: ", accuracy)
    for key, values in scores.items():
            print(model_name+" "+ key,' mean ', values.mean())
            print(model_name+" " + key,' std ', values.std())

print("-----------------------------------------------------------------")
print("Best Model is " + best_model)
print("Best Accuracy: ", best_acc)
print("Best score mean: ", best_score)


#find best parameters
best_model = ""
best_acc = 0
best_sc = 0
print("Tuning hyperparameters for the models:..............................................")
for classifier in clfs:
    pipeline.set_params(clf=classifier)
    if(isinstance(classifier, LinearSVC) ):
      cv_grid = GridSearchCV(pipeline, param_grid={
      'clf__C': np.linspace(0.01, 1.2, 10) #around 0.406
     }, cv = 3)
    elif (isinstance(classifier, LogisticRegression)  ):
        cv_grid = GridSearchCV(pipeline, param_grid={
            'clf__C': [700, 1000] ,# between 500 to 1000
            'clf__max_iter': [1000000]
        }, cv = 3)
    elif isinstance(classifier, AdaBoostClassifier):
      cv_grid = GridSearchCV(pipeline, param_grid={
      'clf__learning_rate': np.linspace(0.01, 1, 5)#around 0.72
      #'clf__n_estimators': [200,400,1000] # try it after finding best lr
     }, cv = 3)
    elif isinstance(classifier, RandomForestClassifier):
      cv_grid = GridSearchCV(pipeline, param_grid={
      'clf__n_estimators': [200,400,1000] #1000 is best
      #'clf_max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None] #try it later
     }, cv = 3)
    elif isinstance(classifier, DecisionTreeClassifier):
        cv_grid = GridSearchCV(pipeline, param_grid={
            'clf__max_depth': np.linspace(1, 10, 10, endpoint=True)
        }, cv=3)
    else: continue
    cv_grid.fit(train_data, train_target)
    best_par = cv_grid.best_params_
    best_estimator = cv_grid.best_estimator_
    best_score = cv_grid.best_score_
    model_name = type(classifier).__name__
    print("-----------------------------------------------------------------")
    print(model_name)
    print(model_name + " best parameter: " + str(best_par))
    print(model_name + " best estimator: "+ str(best_estimator.steps[2]))
    print(model_name + " best CV score: "+ str(best_score))
    y_predict = cv_grid.predict(test_data)
    accuracy = accuracy_score(test_target, y_predict)
    print('Accuracy of the best ' + model_name + 'after CV is %.6f%%' % (accuracy * 100))
    if (accuracy > best_acc):
        best_acc = accuracy
        best_sc = best_score
        best_model = model_name

print("-----------------------------------------------------------------")
print("After Tuning hyperparameters the best Model is " + best_model)
print("Best Accuracy: ", best_acc)
print("Best score mean: ", best_score)