import math as m
##################################################
#x = chromosome list
#D = Dimension of the list or the no. of Attributes
##################################################
def fitnessFunction(x,D):
    sum_of_sq = sum(i**2 for i in x)
    sum_of_cos = sum(m.cos(2*3.14*i) for i in x)

    return round((-20*m.exp(-0.02*(m.sqrt(sum_of_sq/D))))-(m.exp(sum_of_cos/D) ),2) + (20 + m.exp(1))
    
    
    
