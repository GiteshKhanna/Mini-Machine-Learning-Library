##
#####trainingSet
#####dependentSolSet
#####no_of_attributes: Number of parameters
##

##Notes: in numpy array. arr1.dot(arr2)-- Matrix Multiplication
                    #### arr1*arr2-- Element wise Multiplication

import numpy as np


class Linear_Regression:
    def __init__(self,trainingSet,dependentSolSet):
        self.trainingSet=np.array(trainingSet)
        self.dependentSolSet=np.array(dependentSolSet)
        self.no_of_attributes = 0
        
        for i in range(len(trainingSet)):
            self.no_of_attributes = max(self.no_of_attributes,len(trainingSet[i]))
            
        self.theta = np.array([1 for i in range(self.no_of_attributes)])

    def hypothesis(self):
        return (self.trainingSet.dot(self.theta))


##x1*theta1 + x2*theta2 + x3*theta3 + ....
    def costFunction(self):
        h = self.hypothesis()
        sub = np.subtract(self.dependentSolSet,h)
        return (sum(sub**2))/(2*len(sub))

    def makeModel(self,learning_rate=0.2):
        costf_old = self.costFunction()
        y = self.dependentSolSet
        x = self.trainingSet
        m = len(y)
        iter = 0

        while(iter!=1):
            
            delta=np.array(self.hypothesis()-y)

            for i in range(len(self.theta)):
                self.theta[i]= self.theta[i] - (learning_rate*(sum(delta*x[:,i])))/m

        
            print(self.theta)
            iter+=1
            
            

        
                
##Making theta
        

       
            

    def predict(self,testSet):
        return (self.testSet.dot(self.theta))

'''
##Testing
object = Linear_Regression([[2,3],[5,6],[10,5]],[244,988,544])
object.makeModel()
h =object.hypothesis()
y = object.dependentSolSet
x = object.trainingSet
makingmod = object.makeModel()
'''




