import numpy as np
from sklearn.datasets import fetch_20newsgroups
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
import matplotlib.pyplot as plt
import pandas as pd



# Get training data from the 20 news group dataset
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, remove=(['headers', 'footers', 'quotes']))
#print(twenty_train.target_names)

#plot dataset(a problem with dataset)
'''
df = pd.DataFrame(twenty_train.data, columns=twenty_train.feature_names)
fig = plt.figure(figsize=(8,6))
df.groupby('target_names').Consumer_complaint_narrative.count().plot.bar(ylim=0)
plt.show()
'''

# Tokenize text by indexing each word in each document and associate it with its occurrence
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

# Transform occurrence into tf-idf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#test set
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
docs_test = twenty_test.data

#fit data with different models using pipelines and forloop

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


'''
print("Test for different models:................................................")
for classifier in clfs:
    pipeline.set_params(clf = classifier)
    pipeline.fit(twenty_train.data, twenty_train.target)
    predicted = pipeline.predict(docs_test)
    accuracy = np.mean(predicted == twenty_test.target)
    scores = cross_validate(pipeline,twenty_train.data, twenty_train.target)
    model_name = type(classifier).__name__
    print("-----------------------------------------------------------------")
    print(model_name)
    print(model_name+" accuracy: ", accuracy)
    for key, values in scores.items():
            print(model_name+" "+ key,' mean ', values.mean())
            print(model_name+" " + key,' std ', values.std())

'''

#find best parameters
print("Tuning hyperparameters for the models:..............................................")
for classifier in clfs:
    pipeline.set_params(clf=classifier)
    if(isinstance(classifier, LinearSVC) ):
      cv_grid = GridSearchCV(pipeline, param_grid={
      'clf__C': np.linspace(0.01, 1.5, 3)

     })
    elif (isinstance(classifier, LogisticRegression)  ):
        cv_grid = GridSearchCV(pipeline, param_grid={
            'clf__C': [0.1,1,10,100,1000] ,# logistic regression has a problem
            'clf__max_iter': [4000]
        })
    elif isinstance(classifier, AdaBoostClassifier):
      cv_grid = GridSearchCV(pipeline, param_grid={
      'clf__learning_rate': np.linspace(0.1, 1.5, 3)
     })
    elif isinstance(classifier, RandomForestClassifier):
      cv_grid = GridSearchCV(pipeline, param_grid={
      'clf__n_estimators': np.linspace(50, 200, 3)
     })
    else: continue
    cv_grid.fit(twenty_train.data, twenty_train.target)
    best_par = cv_grid.best_params_
    best_estimator = cv_grid.best_estimator_
    best_score = cv_grid.best_score_
    model_name = type(classifier).__name__
    print("-----------------------------------------------------------------")
    print(model_name)
    print(model_name + " best parameter: " + str(best_par))
    print(model_name + " best estimator: "+ str(best_estimator.steps[2]))
    print(model_name + " best CV score: "+ str(best_score))
    y_predict = cv_grid.predict(docs_test)
    accuracy = accuracy_score(twenty_test.target, y_predict)
    print('Accuracy of the best ' + model_name + 'after CV is %.6f%%' % (accuracy * 100))

