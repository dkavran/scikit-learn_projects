
# coding: utf-8

# In[1]:
from sklearn.metrics import  classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split


# In[2]:

from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
mnist

X, y = mnist["data"], mnist["target"]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000],y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# In[3]:

classificators = [None] * 10

Cs = [0.001, 0.01, 0.1, 1, 10]
kernels = ['linear']
probability = [True]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=6000, test_size=1000, random_state=42)
#smaller train and test sets to speed up the process - consequently smaller precision and recall
for index in range(10):
    y_train_tmp = (y_train == index)
    svc = SVC()

    parameters = {'C': Cs,'kernel':kernels, 'probability':probability}

    grid_search = GridSearchCV(svc, parameters, n_jobs=-1, cv=3)
    grid_search.fit(X_train, y_train_tmp)

    svc = grid_search.best_estimator_
    classificators[index] = svc


# In[4]:

probabilities = [None] * 10
for index, classificator in enumerate(classificators):
    probabilities[index] = classificator.predict_proba(X_test)[:,1]
probabilities = np.asarray(probabilities)
predictions = probabilities.argmax(axis=0)


# In[6]:

print(classification_report(predictions,y_test),"\n",confusion_matrix(y_test,predictions)) #around 0.86 both precision and recall
