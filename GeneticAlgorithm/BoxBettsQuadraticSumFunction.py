import math as m
##################################################
#x = chromosome list
#D = Dimension of the list or the no. of Attributes
##################################################
def fitnessFunction(x,D):
    s = 0
    for i in range(D-1):
        s+= m.exp(-0.1*(i+1)*x[0]) - m.exp(-0.1*(i+1)*x[1]) - m.exp(((-0.1*(i+1))-m.exp(-(i+1)))*x[2])
        

    return s
    
    
    
