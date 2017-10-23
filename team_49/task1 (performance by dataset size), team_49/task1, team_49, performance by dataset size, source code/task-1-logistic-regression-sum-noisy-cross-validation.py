
# coding: utf-8

# In[103]:

get_ipython().magic(u'matplotlib inline')


# In[104]:

import pandas as pd


# In[105]:

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


# In[106]:

import config as cfg
sum_without_noise = cfg.DATA_SETS['sum_without_noise']
sum_with_noise = cfg.DATA_SETS['sum_with_noise']


# In[107]:

df = pd.read_csv(sum_with_noise, sep=';')


# In[108]:

df = df.head(500000)


# In[109]:

# label encoding
df['Noisy Target Class (Encoded)'] = df['Noisy Target Class'].astype('category')
df['Noisy Target Class_codes'] = df['Noisy Target Class (Encoded)'].cat.codes


# In[110]:

X = df.drop(['Noisy Target', 'Noisy Target Class', 'Noisy Target Class (Encoded)', 'Noisy Target Class_codes'], axis=1)
X = X.drop('Feature 5 (meaningless)', axis=1)


# In[119]:

y = df[['Noisy Target Class_codes']]
np.shape(y)


# In[78]:

# Split X and y into X_
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70/30 split
# regression_model = LinearRegression()
regression_model = linear_model.LogisticRegression()
regression_model.fit(X_train, y_train)


# In[ ]:

# Cross Validation (10 fold)
regression_model = linear_model.LogisticRegression()
scores = cross_val_score(regression_model, X, np.squeeze(y), cv=10) # np.squeeze required as (a,) needed instead of (a,1)

# The mean score and the 95% confidence interval of the score estimate are hence given by
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[62]:

for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))


# In[63]:

intercept = regression_model.intercept_[0]

print("The intercept for our model is {}".format(intercept))


# In[64]:

regression_model.score(X_test, y_test)


# In[65]:

y_predict = regression_model.predict(X_test)

regression_model_mse = mean_squared_error(y_predict, y_test)

math.sqrt(regression_model_mse)


# In[110]:

# plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_predict, color='blue', linewidth=1)
plt.plot(X_test, y_test, color='yellow', linewidth=0.25)

# plt.plot(y_test, y_predict, color = 'blue')


plt.show()


# In[115]:

# Cross Validation (10 fold)
regression_model = regression_model = linear_model.LogisticRegression()
scores = cross_val_score(regression_model, X, np.squeeze(y), cv=10)


# In[117]:

scores


# In[118]:

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:



