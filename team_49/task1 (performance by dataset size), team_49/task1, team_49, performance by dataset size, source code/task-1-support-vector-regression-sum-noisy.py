
# coding: utf-8

# In[2]:

get_ipython().magic(u'matplotlib inline')


# In[3]:

import pandas as pd


# In[7]:

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt


# In[8]:

import config as cfg
sum_without_noise = cfg.DATA_SETS['sum_without_noise']
sum_with_noise = cfg.DATA_SETS['sum_with_noise']


# In[9]:

df = pd.read_csv(sum_with_noise, sep=';')


# In[10]:

df = df.head(50000)


# In[11]:

df = df.drop('Noisy Target Class', axis = 1)


# In[12]:

X = df.drop('Noisy Target', axis=1)
X = df.drop('Feature 5 (meaningless)', axis=1)


# In[13]:

y = df[['Noisy Target']]


# In[15]:

n_samples, n_features = 10, 5
np.random.seed(0)
y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)
X.shape, y.shape


# In[66]:

# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[16]:

# Split X and y into X_
# This is the 70/30 split set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
clf = svm.SVR(C=1.0, epsilon=0.2)
clf.fit(X_train, y_train)


# In[ ]:

clf = svm.SVR(C=1.0, epsilon=0.2)
scores = 


# In[17]:

clf.score(X_test, y_test)


# In[18]:

y_predict = clf.predict(X_test)

regression_model_mse = mean_squared_error(y_predict, y_test)

regression_model_mse

math.sqrt(regression_model_mse)


# In[19]:

# plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_predict, color='blue', linewidth=1)
plt.plot(X_test, y_test, color='red', linewidth=0.25)


plt.show()


# In[ ]:



