import math as m
##################################################
#x = chromosome list
#D = Dimension of the list or the no. of Attributes
##################################################
def fitnessFunction(x,D):
    first = (x[1]-((5.1*(x[0])**2)/(4*((3.14)**2)))+ 5*x[0]/3.14 - 6)**2
    sec = 10*(1-(1/8*3.14))*m.cos(x[0]) + 10

    return first + sec
    
    
    
