import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
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
from sklearn.datasets import fetch_20newsgroups

twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)

print(twenty_train.target_namesg)

# Tokenize text by indexing each word in each document and associate it with its occurrence
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

# Transform occurrence into tf-idf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#test set
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
docs_test = twenty_test.data

#pipeline for logistic regression
text_logreg = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
  ('logreg', LogisticRegression(solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000))
])

text_logreg.fit(twenty_train.data, twenty_train.target)
predicted = text_logreg.predict(docs_test)
accuracy = np.mean(predicted == twenty_test.target)

print("Logistic Regression accuracy:", accuracy)
#cross validation
scores = cross_validate(text_logreg, twenty_train.data, twenty_train.target)
print("Logistic Regression cross validation:", scores)
print("Logistic Regression cross validation mean score:", np.mean(scores['test_score']))
#Decision tree
text_tree = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
  ('tree', DecisionTreeClassifier())
])
text_tree.fit(twenty_train.data, twenty_train.target)
predicted = text_tree.predict(docs_test)
accuracy = np.mean(predicted == twenty_test.target)
print("Decision Tree Classifier accuracy:", accuracy)


scores = cross_validate(text_tree, twenty_train.data, twenty_train.target)
print("Decision Tree Classifier cross validation:", scores)
print("Decision Tree Classifier cross validation mean score:", np.mean(scores['test_score']))

#SVC
text_svc = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
  ('tree', LinearSVC())
])
text_svc.fit(twenty_train.data, twenty_train.target)
predicted = text_svc.predict(docs_test)
accuracy = np.mean(predicted == twenty_test.target)
print("Linear SVC accuracy:", accuracy)


scores = cross_validate(text_svc, twenty_train.data, twenty_train.target)
print("Linear SVC cross validation:", scores)
print("Linear SVC cross validation mean score:", np.mean(scores['test_score']))

clfs = []
clfs.append(LogisticRegression(random_state= np.random, solver='saga', max_iter=10000,tol=1.e-8)) #c
clfs.append(DecisionTreeClassifier(random_state=np.random))
clfs.append(LinearSVC(random_state=np.random, max_iter=10000))#c
clfs.append(AdaBoostClassifier(random_state=np.random)) #learning rate
clfs.append(RandomForestClassifier(random_state=np.random))# n_estimators