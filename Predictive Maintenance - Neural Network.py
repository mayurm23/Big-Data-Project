
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from keras.layers.core import Flatten


# In[4]:


df = pd.read_csv('Modified-Big data.csv')
df.head()


# In[5]:


df_norm = df[['smart_1_raw','smart_5_raw','smart_9_raw', 'smart_194_raw', 'smart_197_raw','Temp','Duration','RPM']].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_norm.sample(n=5)


# In[7]:


target = df[['failure']].replace(['NOT FAIL','FAIL'],[0,1])
target.sample(n=5)


# In[8]:


df1 = pd.concat([df_norm, target], axis=1)
df1.sample(n=5)


# In[22]:


X = df1.drop(['failure'], axis=1).values
y = df1['failure'].values


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=123)


# In[24]:


model = Sequential()
model.add(Dense(32, input_dim=8, kernel_initializer='normal', activation='relu'))
model.add(Dense(16, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[25]:


print(model.summary())
print(x_train.shape[1])
print(y_train.shape)


# In[26]:


model.fit(x_train, y_train, epochs=, batch_size=50)


# In[15]:


#from keras.models import load_model
#model = load_model('model_3.h5')


# In[27]:


results = model.evaluate(x_test, y_test, verbose=1)


# In[28]:


results


# In[29]:


kfold = KFold(n_splits=3, shuffle=True, random_state=123)
cvresults = []
for train, test in kfold.split(X, y):
    model.fit(X[train], y[train], epochs=10, batch_size=30, verbose=1)
    results = model.evaluate(X[test], y[test], verbose=1)
    cvresults.append(results)


# In[30]:


cvresults


# In[31]:


results = model.evaluate(x_test, y_test, verbose=1)


# In[33]:


results


# In[35]:


from sklearn.metrics import classification_report,confusion_matrix
y_pred = model.predict_classes(x_test)
target_names= ['Class 1(FAIL)', 'Class 0(NOT FAIL)']
print("\n",confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred,target_names=target_names))


# In[37]:


x = [i for i in range(len(y_test))]
plt.scatter(x, y_test, label='test', alpha=1)
plt.scatter(x, y_pred, label='test predictions', color='red', alpha=0.5)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.4)
plt.figure(figsize=(16, 16), dpi=80)
plt.show()

