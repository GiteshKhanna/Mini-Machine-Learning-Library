import math as m
##################################################
#x = chromosome list
#D = Dimension of the list or the no. of Attributes
##################################################
def fitnessFunction(x,D):
    first = (x[0]+10)**2
    sec = (x[1]+10)**2
    third = m.exp((-(x[0])**2) - ((x[1])**2))
    return first + sec + third    
    
