'''
NOTE:
Remeber pushing dataset into PCA which is
already mean normalized.
Feature Scaling remains optional
'''

import numpy as np
import matplotlib.pyplot as plt

class PCA:
    #does not take first column as ones.
    def __init__(self,x):
        self.set = np.array(x)
        self.set= self.set.T
        self.sigma = np.cov(self.set)
        self.lastFit = self.set
        self.dimensions = len(self.sigma)
        [self.U,self.S,self.V] = np.linalg.svd(self.sigma)

#Fits the parameters onto the set passed as parameter.
    def transform(self,x):
        self.lastFit=x.T
        return self.getRedDimSet(self.dimensions)

#Returns a reduced dimension set(Rows->Data Examples Columns->Features)
    def getRedDimSet(self,k):
        self.dimensions = k
        Ureduced = self.U[:,0:k]
        z = (Ureduced.T).dot(self.lastFit)
        '''
        Prints Reconstructed Matrix after dimension reduction
        print(Ureduced.dot(z).T)
        print(" ")
        '''
        if(self.lastFit!=self.set):
            self.lastFit=self.set
        
        return z.T

        
#A method which reduces dimensions to 2. And plots. Used to Visualize data.
    def visData(self,x1='x1',x2='x2'):
        visSet = self.getRedDimSet(2)
        visSet = visSet.T
        plt.scatter(visSet[:,0],visSet[:,1],color='red')
        plt.title('Visualizing Data in 2D')
        plt.xlabel(x1)
        plt.ylabel(x2)
        plt.show()

#Returns a dimension suited best for the variance tolerable
        '''
    def getDim(self,var=0.05):
        #Since sigma = nxn matrix where n = no. of features
        n = len(self.sigma)
        wholeSum = sum(self.S)
        for k in range(1,n):
            print(k)
            num = sum(self.S[i] for i in range(0,k))
            print("num",num,"wholeSUm",wholeSum)
            print("Calc",(num/wholeSum))
            if(1.0 -(num/wholeSum)<=var):
                break
        
        if(self.lastFit!=self.set):
            self.lastFit=self.set
        print(k)
        return k
        '''

    def getDim(self,var=0.01):
        n = len(self.sigma)
        for k in range(1,n):
            z = self.getRedDimSet(k)
            reconstructedSet = (self.U[:,0:k]).dot(z.T).T
            calc = (np.linalg.norm(self.set.T - reconstructedSet)**2/(np.linalg.norm(self.set.T))**2)
            #print(calc)
            if( calc <=var):
                break
        
               
        if(self.lastFit!=self.set.T):
            self.lastFit=self.set       

        #print(k)
        self.dimensions = k
        return [k,z]

    def autoCompress(self,var=0.01):
        [k,z] = self.getDim(var)
        self.dimensions = k
        if(self.lastFit!=self.set):
            self.lastFit=self.set
        return z
    
        

    

        
            

            
            
        

        
        
        
        
        
        
        
        
        
        
