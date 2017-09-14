
# coding: utf-8

# In[1]:


import numpy as np
import requests
import tarfile, os
import codecs, re, string

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


class ReplaceURLs(BaseEstimator, TransformerMixin):
    def __init__(self, replace = True):
        self.replace = replace
    def fit(self, X_, y=None):
        return self
    def transform(self, X_, y=None):
        if self.replace:
            for index, email in enumerate(X_):
                X_[index] = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', 'URL ', email)
        return X_
    
class ReplaceNumbers(BaseEstimator, TransformerMixin):
    def __init__(self, replace = True):
        self.replace = replace
    def fit(self, X_, y=None):
        return self
    def transform(self, X_, y=None):
        if self.replace:
            for index, email in enumerate(X_):
                X_[index] = re.sub('\d', 'NUMBER ', email)
        return X_


# In[3]:


#dataset link: https://spamassassin.apache.org/old/publiccorpus/
spam_filename = "20021010_spam.tar.bz2"
ham_filename = "20021010_easy_ham.tar.bz2"


tar = tarfile.open(spam_filename) #extracting tar.bz2 files on jupyter
tar.extractall()
tar.close()

tar = tarfile.open(ham_filename)
tar.extractall()
tar.close()


# In[4]:


spam_folder = "spam"
ham_folder = "easy_ham"

spam_array = []
ham_array = []

for filename in os.listdir(spam_folder): #reading emails
    
    with codecs.open(spam_folder+"/"+filename, "r",encoding='utf-8', errors='ignore') as f:
        data = f.read().replace('\n',' ')
        spam_array.append(data)
        
for filename in os.listdir(ham_folder):
    
    with codecs.open(ham_folder+"/"+filename, "r",encoding='utf-8', errors='ignore') as f:
        data = f.read().replace('\n',' ')
        ham_array.append(data)


# In[5]:


X = np.concatenate([spam_array, ham_array])


# In[6]:


y = np.full((1,X.shape[0]),1) #all spam y's get 1s
y[0][len(spam_array):] = 0 #analogue for ham y's
y = np.ravel(y)


# In[7]:


pipeline = Pipeline([
    ('ReplaceURLs',ReplaceURLs(replace=True)),
    ('ReplaceNumbers',ReplaceNumbers(replace=True)),
    ('CountVectorizer',CountVectorizer(lowercase=True))
])

X = pipeline.fit_transform(X)


# In[8]:


shuffle_index = np.random.permutation(X.shape[0]) #used to shuffle both X and y matrices
X = X[shuffle_index]
y = y[shuffle_index]

X_train, X_test = X[:2400], X[2400:] #around 80:20 ratio selection of train and test sets
y_train, y_test = y[:2400], y[2400:]


# In[9]:


models = [KNeighborsClassifier(), RandomForestClassifier(), DecisionTreeClassifier(), SGDClassifier()]
results = [None] * 4
for index, model in enumerate(models):
    models[index] = model.fit(X_train, y_train)


# In[10]:


for model in models:
    prediction_train = cross_val_predict(model, X_train, y_train, cv=3)
    print("Train data\n",classification_report(prediction_train,y_train),"\n",confusion_matrix(y_train,prediction_train))
    
    prediction_test = model.predict(X_test)
    print("\nTest data\n",confusion_matrix(y_test,prediction_test))
    print("Accuracy: ",accuracy_score(y_test,prediction_test),"\n")


# In[11]:


#model #3 - DecisionTreeClassifier has the best precision/recall ratio, lets train it

dtc = DecisionTreeClassifier()

parameters = {
    'max_depth': list(range(2, 50)),
    'min_samples_split': (2,),
    'min_samples_leaf': (1,)
}

grid_search = GridSearchCV(dtc, parameters, n_jobs=-1, cv=3)
grid_search.fit(X_train, y_train)
dtc = grid_search.best_estimator_


# In[12]:


prediction_train = cross_val_predict(dtc, X_train, y_train, cv=3)
print("Train data\n",classification_report(prediction_train,y_train),"\n",confusion_matrix(y_train,prediction_train))

prediction_test = dtc.predict(X_test)
print("\nTest data\n",confusion_matrix(y_test,prediction_test))
print("\nAccuracy: ",accuracy_score(y_test,prediction_test))

