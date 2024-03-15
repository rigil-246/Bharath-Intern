#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv(r"C:\Users\Acer\Downloads\archive (1)\spam.csv",encoding='latin-1')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.shape


# In[7]:


data.isna().sum()


# In[8]:


data = data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'])


# In[9]:


data.columns


# In[10]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() # creating an object for label encoder


# In[11]:


data['v1'] = le.fit_transform(data['v1'])


# In[12]:


data.v1.value_counts()


# In[13]:


data.duplicated().sum()


# In[14]:


ham = data[data['v1']==0].sample(653)


# In[15]:


ham.head()


# In[16]:


spam = data[data['v1']==1]


# In[17]:


spam.info()


# In[18]:


data2 = pd.concat([ham,spam],axis=0)


# In[19]:


data2.head()


# In[20]:


data2.info()


# In[21]:


data2.duplicated().sum(), data2.isnull().sum()


# In[22]:


pip install wordcloud


# In[23]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[24]:


hamt = data2[data2['v1']==1]
spamt = data2[data2['v1']==0]


# In[25]:


hamt.head()


# In[26]:


ham_txt = " ".join(hamt['v2'])
len(ham_txt)


# In[27]:


wc = WordCloud(width=900,height=800).generate(ham_txt)
plt.imshow(wc)
plt.axis('off')
plt.show()


# In[28]:


spamt.head()


# In[29]:


spam_txt = " ".join(spamt['v2'])
len(spam_txt)


# In[30]:


wc1 = WordCloud(width = 900,height = 800).generate(spam_txt)
plt.imshow(wc1)
plt.axis('off')
plt.show()


# In[31]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(data2['v2'],data2['v1'],test_size=0.2,random_state=42,stratify=data2['v1'])


# In[32]:


xtrain.shape,ytrain.shape,xtest.shape,ytest.shape


# In[33]:


import spacy


# In[34]:


nlp = spacy.load("en_core_web_sm")


# In[35]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[36]:


model = Pipeline([
    ('text preprocessing',TfidfVectorizer(
    tokenizer = lambda txt:[token.lemma_ for token in nlp(txt)],
    stop_words = 'english',
    ngram_range = (1,2),
    max_features = 500)),
    ('text classifier',RandomForestClassifier())
])


# In[37]:


model.fit(xtrain,ytrain)


# In[38]:


ypred = model.predict(xtest)


# In[39]:


ypred[:5]


# In[40]:


from sklearn.metrics import classification_report as cr
print(cr(ytest,ypred))


# In[41]:


sample_txt = 'hurray !! you won a 25 million dollors'


# In[42]:


model = Pipeline([
    ('text preprocessing',TfidfVectorizer(
    tokenizer = lambda txt:[token.lemma_ for token in nlp(sample_txt)],
    stop_words = 'english',
    ngram_range = (1,2),
    max_features = 500)),
    ('text classifier',RandomForestClassifier())
])


# In[43]:


model.fit(xtrain,ytrain)


# In[44]:


ypred = model.predict([sample_txt])


# In[45]:


print(ypred[0])


# In[ ]:




