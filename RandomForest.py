import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
import sys
import warnings
warnings.filterwarnings("ignore")

a=sys.argv
#np.append(a,'0')
for i in range(1,len(a)):
    a[i]=int(a[i])

heart = pandas.read_csv("pc1.csv")
heart = heart.replace("?","0")
heart = heart.replace(np.nan,"0")


column=heart.columns
#column=column[1:]
heart=heart.append(pandas.Series(a[1:],index=column),ignore_index=True)

heart.loc[heart["heartpred"]==2,"heartpred"]=1
heart.loc[heart["heartpred"]==3,"heartpred"]=1
heart.loc[heart["heartpred"]==4,"heartpred"]=1
heart["slope"] = heart["slope"].fillna(heart["slope"].median())
heart["thal"] = heart["thal"].fillna(heart["thal"].median())
heart["ca"] = heart["ca"].fillna(heart["ca"].median())

predictors=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
alg=RandomForestClassifier(n_estimators=75,min_samples_split=40,min_samples_leaf=20)
kf=KFold(heart.shape[0],n_folds=70, random_state=0)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (heart[predictors].iloc[train,:])

    # The target we're using to train the algorithm.
    train_target = heart["heartpred"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(heart[predictors].iloc[test,:])
    predictions.append(test_predictions)
# The predictions are in three separate numpy arrays.  Concatenate them into one.  
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)
# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
i=0
count=0
for each in heart["heartpred"]:
    #print("p",predictions)
    if each==predictions[i]:
        count+=1
    i+=1
accuracy=count/i
print("Random Forest Result:-")
print("Accuracy = ")
print(accuracy*100)
print("predicted value is",predictions[-1])
#print(predictions)
import matplotlib.pyplot as plt
x=[]
for i in range(len(predictions)):
    x.append(i)
fig = plt.figure(figsize=(5, 5))
plt.plot(heart["heartpred"],x)
plt.xlabel('predicted value')
plt.ylabel('data set')
plt.plot(predictions,x)
plt.show()
z=predictions
y=heart["heartpred"]
colors = (1,0,0)
color2=(0,0,0)
area = 1.5*3
fig = plt.figure(figsize=(5, 10))
plt.scatter( z,x, s=area, c=colors, alpha=0.5)
plt.scatter( y,x, s=area, c=color2, alpha=0.5)
plt.ylabel('data set')
plt.xlabel('actual and predicted values scattered')

plt.show()
