import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Linear_Regression:
    def __init__(self,X,Y):
        self.trainingSet=np.c_[np.ones(len(X)),X]
        self.dependentSolSet=np.array(Y)
        self.no_of_attributes = 0
        
        for i in range(len(X)):
            self.no_of_attributes = max(self.no_of_attributes,len(self.trainingSet[i]))
            
        self.theta = np.array([1.00 for i in range(self.no_of_attributes)])

    def classifier(self):
        return (self.trainingSet.dot(self.theta))

    def hypothesis(self):
        z = self.classifier()
        return(1/(1+(np.exp(-z))))


##constant(x0*theta0) + x1*theta1 + x2*theta2 + x3*theta3 + ....
    def costFunction(self):
        h = self.hypothesis()
        sub = np.subtract(h,self.dependentSolSet)
        return (sum(sub**2))/(2*len(sub))

    def makeModel(self,learning_rate=0.02):
        y = self.dependentSolSet
        x = self.trainingSet
        m = len(y)
        iter = 0

        while(iter!=5000):
            
            delta=np.array(self.hypothesis()-y)
            #Updating Theta[0]--(Because Derivative is different)
            self.theta[0]= self.theta[0] - (learning_rate*(sum(delta)))
            #Updating all others except Theta[0]
            for i in range(1,len(self.theta)):
                self.theta[i]= self.theta[i] - (learning_rate*(sum(delta*x[:,i])))/m

            iter+=1
            #print(self.theta)
            #print('_____________________')
            
        
    def predict(self,testSet):
        testSet=np.c_[np.ones(len(testSet)),np.array(testSet)]
        z = testSet.dot(self.theta)
        predictedValues=(1/(1+(np.exp(-z))))
        prediction = []
        for i in predictedValues:
            if i<0.5 :
                prediction.append(0)
            elif i>0.5 :
                prediction.append(1)
        return prediction


##################################################################################
#TESTING
##################################################################################


#Importing dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X= dataset.iloc[: ,[2,3]].values
Y= dataset.iloc[: , 4].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer.fit(X)
X= imputer.transform(X)

#Splitting the dataset into training set and testing set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state = 0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print('Predifined Regressor Solutions:')
print(Y_pred)



##ML
Object = Linear_Regression(X_train,Y_train)
Object.makeModel(learning_rate=0.001)
print('Real Solution:')
print(Y_test)
print('Predicted Solution')
prediction = Object.predict(X_test)
print(prediction)


#Checking Validations using Confusion Matrix
from sklearn.metrics import confusion_matrix
#Checking validation of my Model
cm = confusion_matrix(Y_test, prediction)
#Checking Validation of predifined Model
cm1 = confusion_matrix(Y_test,Y_pred)
        
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, prediction
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

