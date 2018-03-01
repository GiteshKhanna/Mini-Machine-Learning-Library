from objectivefunction import objfunction as ff
import random

path = 'C:\Python\OptimalObjectiveFunctionSolution\dataset.csv'
upperb = 9999
iteration = 200
nop=10 #No. of attributes/parameters
chromosome=[0 for i in range(nop)]

#Heading file write
fop = open(path,'w')
fop.write('Iterations,')
for i in range(nop):
    fop.write('x'+str(i)+',')

fop.write('Solution\n')
#Heading Ends



for i in range(1,iteration+1):
    if(i%10==0):
        
        fop.write(str(i)+',')
        
        for j in range(nop):
            fop.write(str(chromosome[j])+',')
            
        fop.write(str(upperb))
        fop.write('\n')
        
        

    for j in range(nop):
        chromosome = [int(random.uniform(25,100)) for k in range(nop)]
        
    

    if((ff(chromosome)<upperb)):
        upperb = ff(chromosome)



fop.close()
        







    





    



