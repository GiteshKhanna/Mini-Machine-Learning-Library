# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Super set
from sklearn.cross_validation import train_test_split
X_train, X_super, Y_train, Y_super = train_test_split(X, Y, test_size = 0.4, random_state = 0)
# Splitting the SuperSet into Cross Validation set and Test Set
X_test,X_cv,Y_test,Y_cv = train_test_split(X_super,Y_super,test_size = 0.5, random_state = 0)



# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_Y = StandardScaler()
Y_train = sc_Y.fit_transform(Y_train)"""
