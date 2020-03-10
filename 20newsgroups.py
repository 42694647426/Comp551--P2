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
from sklearn.model_selection import cross_validate


# Get training data from the 20 news group dataset
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, remove=(['headers', 'footers', 'quotes']))

print(twenty_train.target_names)


# Tokenize text by indexing each word in each document and associate it with its occurrence
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

# Transform occurrence into tf-idf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

#test set
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
docs_test = twenty_test.data

#pipeline for logistic regression
text_logreg = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
  ('logreg', LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000))
])

text_logreg.fit(twenty_train.data, twenty_train.target)
predicted = text_logreg.predict(docs_test)
accuracy = np.mean(predicted == twenty_test.target)

print("Logistic Regression accuracy:", accuracy)
#cross validation
scores = cross_validate(text_logreg, twenty_train.data, twenty_train.target)
print("Logistic Regression cross validation:", scores)

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

scores = cross_validate(text_logreg, twenty_train.data, twenty_train.target)
print("Decision Tree Classifier cross validation:", scores)