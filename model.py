#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv('merged_data.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:


# Steps

# 0. Preprocess + EDA + Feature Selection
# 1. Extract input and output cols
# 2. Scale the values
# 3. Train test split
# 4. Train the model
# 5. Evaluate the model/model selection
# 6. Deploy the model


# In[7]:


# Define a color mapping dictionary
color_map = {
    'OH': 'red',
    'O2H': 'blue',
}

# Map the categories to colors using the color mapping dictionary
colors = [color_map[category] for category in data['Spin Trap']]

# Create scatter plot with mapped colors
plt.scatter(data['aN'], data['aH'], c=colors)

# Optionally, add color legend
for category, color in color_map.items():
    plt.scatter([], [], color=color, label=category)

plt.legend()
plt.xlim(12, 17)  # Change the x-axis limits from 0 to 6
plt.ylim(0, 26)  # Change the y-axis limits from 0 to 7
plt.xlabel('aN')
plt.ylabel('aH')
plt.show()


# In[8]:


# Define a color mapping dictionary
color_map = {
    'OH': 'red',
    'O2H': 'blue',
}

# Map the categories to colors using the color mapping dictionary
colors = [color_map[category] for category in data['Spin Trap']]

# Create scatter plot with mapped colors
plt.scatter(data['aN'], data['aN/aH'], c=colors)

# Optionally, add color legend
for category, color in color_map.items():
    plt.scatter([], [], color=color, label=category)

plt.legend()
plt.xlim(12, 17)  # Change the x-axis limits from 0 to 6
plt.ylim(0, 2)  # Change the y-axis limits from 0 to 7
plt.xlabel('aN')
plt.ylabel('aN/aH')
plt.show()


# In[9]:


import seaborn as sns
sns.scatterplot(x=data['aN'],y=data['aH'],hue=data['Spin Trap'])
plt.xlim(12, 17)  # Change the x-axis limits from 0 to 6
plt.ylim(0, 26)  # Change the y-axis limits from 0 to 7
plt.xlabel('aN')
plt.ylabel('aH')
plt.show()


# In[10]:


y=data['Spin Trap']
X=data.iloc[:,1:4]


# In[11]:


y


# In[12]:


X


# In[13]:


data.isnull().mean() * 100


# In[14]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(y)


# In[15]:


le.classes_


# In[16]:


y_trf=le.transform(y)


# In[17]:


y_trf


# In[18]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y_trf,test_size=0.2,random_state=2)


# In[19]:


X_train.shape


# In[20]:


from sklearn.impute import KNNImputer
knn = KNNImputer(n_neighbors=3,weights='distance')

X_train_trf = knn.fit_transform(X_train)
X_test_trf = knn.transform(X_test)


# In[21]:


fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(12,5))
ax1.set_title('Before Scaling')
sns.kdeplot(X_train['aN'],ax=ax1)
sns.kdeplot(X_train['aH'],ax=ax2)


# In[22]:


from sklearn.linear_model import LogisticRegression
lor = LogisticRegression()
lor.fit(X_train_trf,y_train)


# In[23]:


y_pred = lor.predict(X_test_trf)


# In[24]:


from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score,f1_score
accuracy_score(y_test,y_pred)


# In[25]:


confusion_matrix(y_test,y_pred)
print("Confusion Matrix\n")
cdf=pd.DataFrame(confusion_matrix(y_test,y_pred),columns=list(range(0,2)))
print(cdf)
print("-"*50)
print("Precision - ",precision_score(y_test,y_pred))
print("Recall - ",recall_score(y_test,y_pred))
print("F1 score - ",f1_score(y_test,y_pred))


# In[26]:


from sklearn.impute import SimpleImputer
si=SimpleImputer()
X_train_trf2 = si.fit_transform(X_train)
X_test_trf2 = si.transform(X_test)


# In[27]:


lor.fit(X_train_trf2,y_train)
y_pred2 = lor.predict(X_test_trf2)


# In[28]:


accuracy_score(y_test,y_pred2)


# In[29]:


print(lor.coef_)
print(lor.intercept_)
coef=lor.coef_
intercept=lor.intercept_


# In[30]:


m1 = -(lor.coef_[0][0]/lor.coef_[0][1])
b1 = -(lor.intercept_/lor.coef_[0][1])


# In[31]:


x_input = np.linspace(12,17,100)
y_input = m1*x_input + b1


# In[32]:


plt.figure(figsize=(10,6))
plt.plot(x_input,y_input,color='black',linewidth=3)
sns.scatterplot(x=X['aN'],y=X['aH'],hue=y)
plt.xlim(12,17)
plt.ylim(0,25)


# In[33]:


a=coef[0][0]
b=coef[0][1]
c=coef[0][2]
d=intercept


# In[34]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

# Calculate x3 for the decision boundary
x1 = np.linspace(X['aN'].min(), X['aN'].max(), 100)
x2 = np.linspace(X['aH'].min(), X['aH'].max(), 100)
X1, X2 = np.meshgrid(x1, x2)
X3 = (-a*X1 - b*X2 + d) / c

# Plot in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X1, X2, X3, alpha=0.5, cmap='viridis')

# Convert class labels to colors using a colormap
cmap = ListedColormap(['red', 'green'])  # You can choose different colors as needed

# Plot scatter plot of original dataset
sc = ax.scatter(X['aN'], X['aH'], X['aN/aH'], c=y_trf, cmap=cmap)

# Labels and title
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
plt.xlim(0,20)
plt.ylim(0,25)
plt.title('Decision Boundary and Original Data')

#plt.colorbar(sc, ticks=[0, 1], label='Class')
#plt.legend()
plt.show()


# In[35]:


knn = KNNImputer(n_neighbors=3,weights='distance')

X_trf = knn.fit_transform(X)


# In[36]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# Initialize cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform cross-validation
scores = cross_val_score(lor, X_trf, y_trf, cv=kfold)

# Print the cross-validation scores
print("Cross-Validation Scores:", scores)
print("Mean CV Score:", scores.mean())


# In[39]:


import joblib
# Save the trained model
joblib.dump(lor, "model.pkl")


# In[ ]:




