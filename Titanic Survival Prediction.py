#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report , accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


data=data = pd.read_csv(r"C:\Users\Acer\Downloads\archive (2).zip")


# In[3]:


data


# In[4]:


data.describe()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data.head()


# In[8]:


data.tail()


# In[9]:


data.isnull().sum()


# In[10]:


data = data.drop(columns = 'Cabin', axis = 1)


# In[11]:


data.head()


# In[12]:


data ['Age'].fillna(data['Age'].mean(), inplace = True)


# In[13]:


data.isnull().sum()


# In[14]:


data ['Fare'].fillna(data['Fare'].mean(), inplace = True)


# In[15]:


data.isnull().sum()


# In[16]:


#Analyzing on how many people have survived
data['Survived'].value_counts()


# In[17]:


#Survived passengers
sns.countplot(x = 'Survived', data = data).set_title
plt.show()


# In[18]:


data['Sex'].value_counts()


# In[19]:


sns.countplot(x = 'Sex', data = data).set_title('Gender')
plt.show()


# In[20]:


#Passengers who have survived the collision
sns.countplot(x = 'Sex', hue = 'Survived', data = data).set_title


# In[21]:


#Number of Passengers in different class
sns.countplot(x = 'Pclass', data = data).set_title


# In[22]:


#Number of Passengers in different class of both Male and Female
sns.countplot(x = 'Pclass', hue = 'Survived', data = data).set_title


# In[23]:


#Encoding catergorical columns into numerical values
data['Sex'].value_counts()


# In[24]:


data['Embarked'].value_counts()


# In[25]:


data.replace({'Sex':{'male':1,'female':2}, 'Embarked':{'S':1,'C':2,'Q':3}}, inplace = True)


# In[26]:


data.head()


# In[27]:


X = data.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Survived'], axis = 1)
Y = data['Survived']
print(X)


# In[28]:


print(Y)


# In[29]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 2)


# In[30]:


print(X.shape, X_train.shape, X_test.shape)


# In[31]:


titanic_test = pd.get_dummies(X, columns=['Sex'], drop_first=True)

#Model Training
#Using Logistic Regression

model = LogisticRegression(random_state = 0)

#Training model
model.fit(X_train, Y_train)


# In[32]:


#Evaluation of the Model
#Accuracy Score
X_train_prediction = model.predict(X_train)


# In[33]:


print(X_train_prediction)


# In[34]:


training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data = ', training_data_accuracy)


# In[35]:


#accuracy on test data
X_test_prediction = model.predict(X_test)
print(X_test_prediction)


# In[36]:


test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data = ', test_data_accuracy)


# In[37]:


print(Y_test)


# In[38]:


X_test_prediction = model.predict(X_test)

# Calculating the survival rate
survival_rate = X_test_prediction.mean()

if survival_rate < 1:
    print("Congratulations! You survived.")
else:
    print("I'm sorry, but you have failed to survive.")


# In[39]:


X=data[["Pclass" , "Sex" , "Age" , "Fare"]]
y=data["Survived"]


# In[40]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X["Sex"] = encoder.fit_transform(X["Sex"])
X


# In[41]:


y


# In[42]:


X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)


# In[43]:


from sklearn.linear_model import LogisticRegression
LR_model=LogisticRegression(max_iter=1500)
LR_model.fit(X_train,y_train)


# In[44]:


LR_pred = LR_model.predict(X_test)


# In[45]:


LR_acc=accuracy_score(y_test,LR_pred)


# In[46]:


print(classification_report(y_test , LR_pred))


# In[47]:


cm1 = confusion_matrix(y_test,LR_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm1,annot=True)
plt.xlabel('predicted')
plt.ylabel('Truh')


# In[48]:


DT_model = DecisionTreeClassifier(random_state=42)
DT_model.fit(X_train , y_train)


# In[49]:


DT_model.score(X_test , y_test)


# In[50]:


DT_pred=DT_model.predict(X_test)


# In[51]:


DT_acc = accuracy_score(y_test ,DT_pred )


# In[52]:


print(classification_report(y_test , DT_pred))


# In[53]:


cm2 = confusion_matrix(y_test,DT_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm2,annot=True)
plt.xlabel('predicted')
plt.ylabel('Truh')


# In[54]:


RF_model = RandomForestClassifier(n_estimators= 100)
RF_model.fit(X_train , y_train)


# In[55]:


RF_model.score(X_train , y_train)


# In[56]:


RF_pred = RF_model.predict(X_test)


# In[57]:


RF_acc= accuracy_score(y_test , RF_pred)  


# In[58]:


print(classification_report(y_test , RF_pred))


# In[59]:


cm3 = confusion_matrix(y_test,RF_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm3,annot=True)
plt.xlabel('predicted')
plt.ylabel('Truh')


# In[60]:


models=pd.DataFrame({
    'models':['Logistic_Regression','Random_forest','Decsion_Treee'],
    'scores':[LR_acc,RF_acc,DT_acc]})
models.sort_values(by='scores',ascending=False)


# In[ ]:





# In[ ]:




