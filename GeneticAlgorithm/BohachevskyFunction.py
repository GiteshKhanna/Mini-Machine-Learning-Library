import math as m
##################################################
#x = chromosome list
#D = Dimension of the list or the no. of Attributes
##################################################
def fitnessFunction(x,D):
    return x[0]**2 + 2*(x[1]**2) - 0.3*m.cos(3*3.14*x[0])-0.4*m.cos(4*3.14*x[1]) +0.7
    
    
    
