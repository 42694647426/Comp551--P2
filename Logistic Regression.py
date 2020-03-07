import sklearn.linear_model
from sklearn.datasets import fetch_20newsgroups

twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)

print(twenty_train.target_namesg)
