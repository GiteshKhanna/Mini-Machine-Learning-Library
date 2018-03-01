# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, 0:3].values
Y = dataset.iloc[:,-1].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
#sc_Y = StandardScaler()
#Y_train = sc_Y.fit_transform(Y)

#Testing PCA
import PCA as pca
A = pca.PCA(X_train)
aSet=A.autoCompress()
bSet=A.transform(X_test)



#Testing Linear Regression on the new Set with my Linear regression model
import myLinearRegression as ff
model = ff.Linear_Regression(aSet,Y_train)
model.makeModel()
prediction = model.predict(bSet)
print("My prediction")
print(prediction)

#Printing Real Y_test
print("Real Solution")
print(Y_test)

#Testing Linear Regression on the new Set with scikit learn Linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(aSet,Y_train)
Y_pred = regressor.predict(bSet)
print('Predefined Regressor Solution:')
print(Y_pred)

#Plotting using scikit Model
plt.scatter(aSet[:,0],Y_train,color='yellow')
plt.plot(bSet[:,0],Y_pred,color='blue')
plt.title('Scikit Model')
plt.xlabel("aSet[:,0]")
plt.ylabel("Result")
plt.show()


plt.scatter(aSet[:,1],Y_train,color='yellow')
plt.plot(bSet[:,1],Y_pred,color='blue')
plt.title('Scikit MODEL')
plt.xlabel("aSet[:,1]")
plt.ylabel("Result")
plt.show()

#Plotting using my Model
plt.scatter(aSet[:,0],Y_train,color='red')
plt.plot(bSet[:,0],prediction,color='blue')
plt.title('My MODEL')
plt.xlabel("aSet[:,0]")
plt.ylabel("Result")
plt.show()

plt.scatter(aSet[:,1],Y_train,color='red')
plt.plot(bSet[:,1],prediction,color='blue')
plt.title('My MODEL')
plt.xlabel("aSet[:,1]")
plt.ylabel("Result")
plt.show()

