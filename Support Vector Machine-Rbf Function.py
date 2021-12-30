import pandas as pd
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
data=pd.read_csv("C:/Users/D E L L/Documents/PROJECT/hungary heart disease data.csv")
print(data)
data = data.replace("?","0")
data = data.replace(np.nan,"0")
y=pd.read_csv("C:/Users/D E L L/Documents/PROJECT/target.csv")
data["thal"]=data["thal"].replace("2","1")
data["thal"]=data["thal"].replace("3","1")
data["thal"]=data["thal"].replace("4","1")
data["thal"]=data["thal"].replace("5","1")
data["thal"]=data["thal"].replace("6","1")
data["thal"]=data["thal"].replace("7","1")
# Split data to train and test on 80-20 ratio
X_train, X_test, y_train, y_test = train_test_split(data,y, test_size = 0.4, random_state=1)
#plt.plot(X_train,y_train)
# Create a linear SVM classifier 
clf = svm.SVC(kernel='linear')
# Create a linear SVM classifier 
clf = svm.SVC(kernel='rbf')
# Train classifier 
clf.fit(X_train, y_train)
# Make predictions on unseen test data
clf_predictions = clf.predict(X_test)
print("Accuracy: {}%".format(clf.score(X_test, y_test) * 100 ))
# Create a linear SVM classifier with C = 1
clf = svm.SVC(kernel='rbf', C=1)
# Grid Search
# Parameter Grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
# Make grid search classifier
clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
# Train the classifier
clf_grid.fit(X_train, y_train)
print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)
x=[]
for i in range(len(y_test)):
    x.append(i)
plt.plot(y_test,x)
plt.plot(X_test["thal"],x)
plt.show()
from sklearn import svm, datasets
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
X, y = make_blobs(n_samples=40, centers=2, random_state=6)
clf.fit(X, y)
# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
