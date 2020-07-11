# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 14:35:54 2020

@author: ramra
"""

# Import the libraries
import numpy as np
import pandas as pd

# Importing the dataset
train_dataset = pd.read_csv('train_20D8GL3.csv')
train_X = train_dataset.iloc[:, :-1].values
dfX = pd.DataFrame(train_X)
train_Y = train_dataset.iloc[:, 24].values
dfY = pd.DataFrame(train_Y)

test_dataset = pd.read_csv('test_O6kKpvt.csv')
test_X = test_dataset.iloc[:, 1].values
dfY_test = pd.DataFrame(test_X)

# Data Preprocessing
# Take care of missing data
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(dfX)
dfX = imputer.transform(dfX)

# Split dataset into Train and Test sets
from sklearn.model_selection import train_test_split
X_train, Y_train, X_val, Y_val = train_test_split(dfX, dfY, test_size=0.20, random_state = np.random.randint(1,1000, 1)[0])

# Remove unwanted features
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
Y_train = pca.transform(Y_train)
explained_variance = pca.explained_variance_ratio_

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
Y_train = sc_X.transform(Y_train)

# Fitting Support Vector Machine to the Training Set
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, X_val)

# Predicting the Test set results
Y_pred = classifier.predict(Y_train)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, X_val)

# Predicting the Test set results
Y_pred = classifier.predict(Y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_val, Y_pred)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()