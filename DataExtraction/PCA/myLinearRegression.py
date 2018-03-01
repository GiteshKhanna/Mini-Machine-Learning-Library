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

    def hypothesis(self):
        return (self.trainingSet.dot(self.theta))


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

        while(iter!=10000):
            
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
        return (testSet.dot(self.theta))
