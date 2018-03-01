import math as m
import random
from copy import deepcopy
#from AckleyFunction import fitnessFunction as ff
#from AlpineFunction import fitnessFunction as ff
#from BartelsConnFunction import fitnessFunction as ff
#from BealeFunction import fitnessFunction as ff
#from BirdFunction import fitnessFunction as ff
#from BohachevskyFunction import fitnessFunction as ff
#from BoothFunction import fitnessFunction as ff
#from BoxBettsQuadraticSumFunction import fitnessFunction as ff
#from BraninRCOSFunction import fitnessFunction as ff
from  BrentFunction import fitnessFunction as ff

#################################################################
#Initializing some parameters
#################################################################

UB = -10  #Upper Bound of random initialization
LB = 10  #Lower Bound of random initialization
D = 2     #Dimension/Attributes
CR = 0.2    #Crossover Rate
MR = 0.01   #Mutation Rate

popSize= 100 #Total chromosomes(Population)
iterations = 10000
pop = []    #A list consisting of the whole population

#################################################################
#Creating an Individual in the population
#################################################################

class Individual:
    def __init__(self,c,f):
        self.chromosome = c
        self.fitness = f
        
#################################################################        
#Initializing randomly created individuals
#################################################################
        
def createChromosomes():
    for i in range(popSize):
        chromosome=[]
        
        for j in range(D):
            
            if j ==0:
                chromosome.append(round(random.uniform(-10,10),2))
            elif j==1:
                chromosome.append(round(random.uniform(-10,10),2))
                '''
            elif j==2:
                chromosome.append(round(random.uniform(0.9,1.2),2))
                '''
            else:
                chromosome.append(round(random.uniform(LB,UB),2))
                
                
            
        fitness = ff(chromosome,D)
        newInd = Individual(chromosome,fitness)
        pop.append(newInd)
                              
        
#################################################################
#Finding the individual with the best Fitness
#################################################################

def getBestIndividual():
    bestIndividual=pop[0]
    for i in pop[1:]:
        if i.fitness<bestIndividual.fitness :
            bestIndividual=i
            
    return bestIndividual

#################################################################
#GA->Crossover
#################################################################
def crossOver():
    #GA-Selection
    #Selecting two random Individuals
    for i in range(popSize):

        if(random.random()<CR):
            ind1,ind2 = random.sample(range(popSize),2)
            p1= deepcopy(pop[ind1])
            p2= deepcopy(pop[ind2])
            

            #Finding Crossover index
            #pt = random.randint(1,D-2)
            pt = 1

            #Crossing1
            c1 = p1.chromosome[0:pt]
            c1.extend(p2.chromosome[pt:])
            #Crossing2
            c2 = p2.chromosome[0:pt]
            c2.extend(p1.chromosome[pt:])

            c1Fitness = ff(c1,D)
            c2Fitness = ff(c2,D)

            # Select between parent and child
            if c1Fitness < p1.fitness:
                pop[ind1].fitness=c1Fitness
                pop[ind1].chromosome=c1
                    
            if c2Fitness < p2.fitness:
                pop[ind2].fitness=c2Fitness
                pop[ind2].chromosome=c2
            

def mutation():
    for i in range(popSize):
        if(random.random()<MR):
            
            #Selection of parent
            r = random.randint(0,popSize-1)
            p = deepcopy(pop[r])

            #Selecting mutation point
            pt = random.randint(0,D-1)

             # Create new child
            c=deepcopy(p.chromosome)

            # Mutation
            if(pt==0):
                c[pt]=round(random.uniform(-10,10),2)
            elif(pt==1):
                c[pt]==round(random.uniform(-10,10),2)
                '''
            elif(pt==2):
                c[pt]=round(random.uniform(0.9,1.2),2)
                '''
            else:
                c[pt] = round(random.uniform(LB,UB),2)
            

            #Get the fitness of childs
            cFitness=ff(c,D)

            
            # Select between parent and child
            if cFitness < p.fitness:
                pop[r].fitness=cFitness
                pop[r].chromosome=c
            

            

            
'''
        #Checking if the child can give a better solution
        #p1,p2  --- c1,c2
        #Case1: Both lowest fitnesses lie on the parent side
        if((p1.fitness<c1Fitness and p2.fitness<c2Fitness) or (p1.fitness<c2Fitness and p2.fitness<c1Fitness)):
            return
        #Case2: Both lowest fitnesses lie on the child side
        elif((p1.fitness>c1Fitness and p2.fitness>c2Fitness) or (p1.fitness>c2Fitness and p2.fitness>c1Fitness)):
            pop[ind1].chromosome=c1
            pop[ind1].fitness = ff(pop[ind1].chromosome,D)
            pop[ind2].chromosome=c2
            pop[ind2].fitness = ff(pop[ind2].chromosome,D)
        #Case3: One low fitness on each side
        else:
            if(p1.fitness>c1Fitness):
                pop[ind1]=c1
                pop[ind1].fitness = ff(pop[ind1].chromosome,D)
            elif(p1.fitness>c2Fitness):
                pop[ind1]=c2
                pop[ind1].fitness = ff(pop[ind2].chromosome,D)
            elif(p2.fitness>c1Fitness):
                pop[ind2]=c1
            elif(p2.fitness>c2Fitness):
                pop[ind2]=c2    
            
'''
####################################################################
#Initializing file to be written into
####################################################################
fp = open('BrentFunction.csv','w')
fp.write('Iterations,fitness,chromosome\n')

createChromosomes()
for k in range(iterations+1):
    bestFit = getBestIndividual()
    crossOver()
    mutation()
    if(k%10==0):
        fp.write(str(k)+','+str(bestFit.fitness)+','+str(bestFit.chromosome)+'\n')
        print(k,')', bestFit.chromosome,bestFit.fitness)
fp.close()
