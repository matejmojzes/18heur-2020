# -*- coding: utf-8 -*-
import visualizer as vs
import itertools
import numpy as np
from timeit import default_timer as timer
import functools as ft

class KnapSack:
    def __init__(self, weights, values, capacity):
        self.weights=weights
        self.values=values
        self.capacity=capacity
        self.numel=len(values)
        
    #Define function __V so that it represents the maximum value we can get under the condition: use first i items, total weight limit is j
    def __V(self, i,j,V):        
        if i == 0 or j <= 0:
            V[i][j] = 0
            return
        #V[i-1[j] has not been calculated, we have to call function __V
        if V[i-1][j]==-1:     
            self.__V(i-1, j,V)
        # item cannot fit in the bag
        if self.weights[i-1] > j:                      
            V[i][j] = V[i-1][j]
        else: 
            #m[i-1,j-w[i]] has not been calculated, we have to call function __V
            if V[i-1][j-self.weights[i-1]] == -1:     
                self.__V(i-1, j-self.weights[i-1],V)
            V[i][j] = max(V[i-1][j], V[i-1][j-self.weights[i-1]] + self.values[i-1])
            
    def dynApproachRec(self):
         V = [[-1 for x in range(self.capacity + 1)] for x in range(self.numel + 1)]
         self.__V(self.numel,W,V)
         resultIdx=list()
         self.__getIdx(V,self.numel,self.capacity,resultIdx)
         weights=tuple(self.__getWeights(resultIdx))
         v=vs.Visualizer(self.capacity)
         v.stackBarChart(weights, [self.values[i] for i in resultIdx],'Optimal solution')
         return {'best_value': V[self.numel][self.capacity], 
                 'result_idx':tuple(resultIdx), 'result_weights': weights}
        
    #getting the indices of optimal subset
    def __getIdx(self,V,i,j, result):
        if i == 0:
            return
        if V[i][j] > V[i-1][j]>=0:
            result.append(i-1)
            self.__getIdx(V,i-1, j-self.weights[i-1],result)
        else:
            self.__getIdx(V,i-1, j,result)
            
    #method finding solution using dynamic programming 
    def dynamicApproach(self):
        #init with zeros an array of n times capacity
        V = [[0 for x in range(self.capacity + 1)] for x in range(self.numel + 1)]
        # Build table V[][] with #numel+1 rows and #capacity columns
        for i in range(self.numel + 1):
            for w in range(self.capacity + 1):
                #for i or w zero, the cell should be zero as well
                if i == 0 or w == 0:
                    V[i][w] = 0
                #else if we have remaining capacity, 
                #we take the maximum - either including ith element, or not
                elif self.weights[i-1] <= w:
                    V[i][w] = max(self.values[i-1]
                              + V[i-1][w-self.weights[i-1]], 
                                  V[i-1][w])
                #else we just copy the previous value
                else:
                    V[i][w] = V[i-1][w]
        #now get the result from this table
        resultIdx=list()
        self.__getIdx(V,self.numel,self.capacity,resultIdx)
        weights=tuple(self.__getWeights(resultIdx))
        v=vs.Visualizer(self.capacity)
        v.stackBarChart(weights, [self.values[i] for i in resultIdx],'Optimal solution')
        
        #return the V[n][W] which is the solution
        return {'best_value': V[self.numel][self.capacity], 'result_idx':tuple(resultIdx), 'result_weights': weights}

   
    #n is the index of an item being included or not
    #bruteforce approach using backtracking
    def __bruteForceRec(self,W,n):
        # initial conditions
        if n == 0 or W == 0 :
            return 0
        # If weight is higher than capacity then it is not included
        if (self.weights[n-1] > W):
            return self.__bruteForceRec(W,n-1)
        # return either nth item being included or not
        else:
            return max(self.values[n-1] + self.__bruteForceRec(W-self.weights[n-1], n-1),
                       self.__bruteForceRec(W,n-1))
        
    #public function for handeling recursion
    def bruteForce(self):
        bestVal=self.__bruteForceRec(self.capacity, self.numel)
        return {'best_value':bestVal}
    
    #get weights of the optimal subset based on its idx
    def __getWeights(self, idx):
            arr = np.array(self.weights)
            return arr[np.array(idx)]
        
    #going over all combinations of all sizes and trying to find solution
    def combinationMethod(self):
        k=1
        currMax=0
        result=list()
        possibilities=list()
        #generate all combinations of all lenghts
        #loop over possible lengths
        while k<= self.numel:
            comb=list(itertools.combinations(range(0,self.numel),k))
            j=0
            #loop over all combinations of given length
            while j<len(comb):
                #get weights of this combination
                weights=[self.weights[i] for i in comb[j]]
                currWeight=sum(weights)
                #if bigger than capacity, continue to next iteration
                if currWeight>self.capacity:
                    j+=1
                    continue
                #get values of this combination
                values=[ self.values[i] for i in comb[j]]
                currVal=sum(values)
                #if better value, save it as currently best solution
                if currVal > currMax:
                    currMax=currVal
                    result=comb[j]
                #keep track of all possibilities
                possibilities.append(comb[j])
                j+=1
            k+=1
        possWeights, possValues=self.__getPossibilities(possibilities)
        v=vs.Visualizer(self.capacity)
        v.barChart(None, wt,val,None, None,'Available items')
        v.stackBarChart(possWeights, possValues, 'Possible solutions')
        weights=tuple(self.__getWeights(result))
        v.stackBarChart(weights, [self.values[i] for i in result],'Optimal solution')
        return {'best_value': currMax, 'result_idx':result, 'result_weights': weights}
      
        
    #generates weights and values from given matrice of indices of possible solutions
    def __getPossibilities(self,idx):
        possWeights=list()
        possValues=list()
        #loop over rows
        for i in idx:
            #get weights and values for these indices and append it to the current matrix
            possWeights.append([self.weights[j] for j in i])
            possValues.append([self.values[j] for j in i])
        return (possWeights, possValues)
    
    # 1. choose the item that has the maximum value from the remaining items;
    # this increases the value of the knapsack as quickly as possible.
    def greedyMaxValue(self):
        #remaining capacity
        W=self.capacity
        values=self.values.copy()
        resultIdx=list()
        #lowest value to replace out values
        _min=min(values)-1
        while True:
           #get index of item with max value
           maxIdx = values.index(max(values))
           #if all  values have been replaced, break loop
           if values[maxIdx] is _min:
               break
           #if possible to add to knapsack
           if self.weights[maxIdx]<=W:
               resultIdx.append(maxIdx)
               W=W-self.weights[maxIdx]
           #remove it as an option
           values[maxIdx]=_min
        weights=tuple(self.__getWeights(resultIdx))
        v=vs.Visualizer(self.capacity)
        v.stackBarChart(weights, [self.values[i] for i in resultIdx],'Suboptimal solution - Max value heuristics')
        #sum best value
        bestVal=np.array(self.values)[resultIdx].sum()
        return {'best_value': bestVal, 'result_idx':tuple(resultIdx), 'result_weights': weights} 
        
    # 2. Choose the lightest item from the remaining items which uses up capacity as slowly 
    # as possible allowing more items to be stuffed in the knapsack.
    def greedyMinWeight(self):
          #remaining capacity
          W=self.capacity
          weights=self.weights.copy()
          resultIdx=list()
          #value higher than capacity
          _max=self.capacity+1
          while True:
             #get index of item with max value
             minIdx = weights.index(min(weights))
             #if all  values have been replaced, break loop
             if weights[minIdx] is _max:
                 break
             #if possible to add to knapsack
             if self.weights[minIdx]<=W:
                 resultIdx.append(minIdx)
                 W=W-self.weights[minIdx]
             #remove it as an option
             weights[minIdx]=_max
          weights=tuple(self.__getWeights(resultIdx))
          v=vs.Visualizer(self.capacity)
          v.stackBarChart(weights, [self.values[i] for i in resultIdx],'Suboptimal solution - Min weight heuristics')
          #sum best value
          bestVal=np.array(self.values)[resultIdx].sum()
          return {'best_value': bestVal, 'result_idx':tuple(resultIdx), 'result_weights': weights} 
      
     # 3. Choose the items with as high a value per weight as possible.  

    def greedyMaxRatio(self):
          #remaining capacity
          W=self.capacity
          #get the ratios
          ratios=[i / j for i, j in zip(self.values, self.weights)]
          resultIdx=list()
          #min value
          _min=min(ratios)-1
          while True:
             #get index of item with max value
             maxIdx = ratios.index(max(ratios))
             #if all  values have been replaced, break loop
             if ratios[maxIdx] is _min:
                 break
             #if possible to add to knapsack
             if self.weights[maxIdx]<=W:
                 resultIdx.append(maxIdx)
                 W=W-self.weights[maxIdx]
             #remove it as an option
             ratios[maxIdx]=_min
          weights=tuple(self.__getWeights(resultIdx))
          v=vs.Visualizer(self.capacity)
          v.stackBarChart(weights, [self.values[i] for i in resultIdx],'Suboptimal solution - Max value/weight ratio heuristics')
          #sum best value
          bestVal=np.array(self.values)[resultIdx].sum()
          return {'best_value': bestVal, 'result_idx':tuple(resultIdx), 'result_weights': weights} 
    
    #genetic algorithm heuristics
    def geneticAlg(self,fitness=None, popCount=50, itCount=300,crossRate=0.9,mutRate=None,prob=0.2):
          #use instance of EvolutionHeur
          evolution=EvolutionHeur(self)
          resultIdxBool, bestVal=evolution.geneticAlg(fitness, popCount,itCount, crossRate, mutRate, prob)
          if type(resultIdxBool) is not int:
              resultIdx=[i for i, e in enumerate(resultIdxBool) if e != 0]
              weights=tuple(self.__getWeights(resultIdx))
              bestVal=np.array(self.values)[resultIdx].sum()
              v=vs.Visualizer(self.capacity)
              v.stackBarChart(weights, [self.values[i] for i in resultIdx],'Suboptimal solution - Genetic evolution heuristics')
          else: 
              resultIdx=[]
              weights=[]
              bestVal=0
          #sum best value
          return {'best_value': bestVal, 'result_idx':tuple(resultIdx), 'result_weights': weights} 

#class for genetic evolution algorithms
class EvolutionHeur:
    #argument being a KnapSack instance
    def __init__(self,Knapsack):
        self.weights=Knapsack.weights
        self.values=Knapsack.values
        self.capacity=Knapsack.capacity
        self.numel=Knapsack.numel

    #fitness function that we want to optimize
    def __fitness(self,x):
      currVal=np.array(x).dot(np.array(self.values).transpose())
      currWeight=np.array(x).dot(np.array(self.weights).transpose())
      if currWeight > self.capacity:
        return 0
      else:
        return currVal
    
    # tournament selection
    def __selection(self,pop, scores, k=3):
        # first random selection
        selectionIdx = np.random.randint(len(pop))
        for idx in np.random.randint(0, len(pop), k-1):
            # check if better (e.g. perform a tournament)
            if scores[idx] < scores[selectionIdx]:
                selectionIdx = idx
        return pop[selectionIdx]
    
    # crossover two parents to create two children
    def __crossover(self,p1, p2, crossRate):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if np.random.rand() < crossRate:
            # select crossover point that is not on the end of the string
            pt = np.random.randint(1, len(p1)-2)
            # perform crossover
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        return [c1, c2]
        
    #mutation operator
    def __mutation(self, bitstring, mutRate):
        for i in range(len(bitstring)):
            # check for a mutation
            if np.random.rand() < mutRate:
                # flip the bit
                bitstring[i] = 1 - bitstring[i]
    
    #heuristics using genetic algorithms
    def geneticAlg(self,fitness=None, popCount=50, itCount=300,crossRate=0.9,mutRate=None,prob=0.2):
        #if no fitness function selected, add default one
        if fitness is None:
            fitness = self.__fitness
        #if no mutation parameter
        if mutRate is None:
            mutRate=1/float(self.numel)
        # initial population of random bitstring with probability of choosing the item
        pop = [[np.random.choice(np.arange(0,2), p=[1-prob,prob]) 
                for _ in range(self.numel)] for _ in range(popCount)]   
        #pop = [np.random.randint(0, 2, self.numel).tolist() for _ in range(popCount)]         
        # keep track of best solution
        bestIdx, bestVal = 0, fitness(pop[0])
        # enumerate generations
        for gen in range(itCount):
            # evaluate all candidates in the population
            scores = [fitness(c) for c in pop]
            # check for new best solution
            for i in range(popCount):
                if scores[i] > bestVal:
                    bestIdx, bestVal = pop[i], scores[i]
                   # print(">generation: %d, new best total value(%s) = %d" % (gen,  pop[i], scores[i]))
            # select parents
            selected = [self.__selection(pop, scores) for _ in range(popCount)]
            # create the next generation
            children = list()
            for i in range(0, popCount, 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i+1]
                # crossover and mutation
                for c in self.__crossover(p1, p2, crossRate):
                    # mutation
                    self.__mutation(c, mutRate)
                    # store for next generation
                    children.append(c)
            # replace population
            pop = children
        #get the results
        return bestIdx, bestVal
    
#example from wikipedia
# val = [4,2,2,1,10]
# wt = [12,2,1,1,4]
# W = 15

# #another one
# val = [50,100,150,200]
# wt = [8,16,32,40]
# W = 64

# #and another
# val = [5,60, 100, 120,1]
# wt = [25,10, 20, 30,40]
# W = 50

# #big sample data
# val=[ 135,139,149,150,156,163,173,184,192,201,210,214,221,229,240]
# wt=[70,73,77, 80, 82, 87, 90, 94, 98,106,110,113,115,118,120]
# W=750

#bigger sample random data
# val=np.random.randint(1,101,30).tolist()
# wt=np.random.randint(1,101,30).tolist()
# W=800

# KnapSackInstance=KnapSack(wt,val,W)

## solutions
# start = timer()
# print(KnapSackInstance.combinationMethod())
# print('Combination method took: %.4f seconds' % float(timer() - start))
# start=timer()
# print(KnapSackInstance.bruteForce())
# print('Brute force took: %.4f seconds' % float(timer() - start))
# start=timer()
# print(KnapSackInstance.dynamicApproach())
# print('Dynamic approach took: %.4f seconds' % float(timer() - start))
# start=timer()
# print(KnapSackInstance.dynApproachRec())
# print('Modified dynamic approach took: %.4f seconds' % float(timer() - start))

##heuristics
# v=vs.Visualizer(W)
# v.barChart(None, wt,val,None, None,'Available items')


# print('Optimal solution: ',KnapSackInstance.dynApproachRec())
# print('Max value heuristics: ', KnapSackInstance.greedyMaxValue())
# print('Min weight heuristics: ', KnapSackInstance.greedyMinWeight())
# print('Max value/weight ratio heuristics: ', KnapSackInstance.greedyMaxRatio())

# print('Genetic alg. heuristics: ', KnapSackInstance.geneticAlg())


#genetic algs comparison
#get optimal value
# opt=KnapSackInstance.bruteForce()['best_value']
# bestTime=np.inf
# bestVal=np.inf
# settingsT=[]
# settingsV=[]
# for i in range(30, 150, 30):
#     print(i)
#     for j in range(50, 1100, 150):
#         for k in range(6, 10):
#             for l in range(0,3):
#                 for m in range(1,3):
#                     start = timer()
#                     curr=KnapSackInstance.geneticAlg(popCount=i, itCount=j,crossRate=k/10,mutRate=l/10,prob=m/10)['best_value']
#                     currTime=timer()-start
#                     currVal=opt-curr
#                     if bestTime>currTime:
#                         bestTime=currTime
#                         settingsT=[ i,j,k/10,l/10,m/10]
#                     if bestVal>currVal:
#                         bestVal=currVal
#                         settingsV=[ i,j,k/10,l/10,m/10]
# print('time-wise',settingsT, bestTime)
# print('accuracy-wise',settingsV, bestVal)

    

##heuristics comparison
# loops=100
# timeResults={"maxVal":np.empty(loops),"minWeight":np.empty(loops),
#               "maxRatio":np.empty(loops),"geneticAlg - default":np.empty(loops),
#               "geneticAlg - default extra iterations":np.empty(loops), 
#               "geneticAlg - time":np.empty(loops),"geneticAlg - accuracy":np.empty(loops)}
# distanceResults={"maxVal":np.empty(loops),"minWeight":np.empty(loops),
#               "maxRatio":np.empty(loops),"geneticAlg - default":np.empty(loops),
#               "geneticAlg - default extra iterations":np.empty(loops), 
#               "geneticAlg - time":np.empty(loops),"geneticAlg - accuracy":np.empty(loops)}
# totalTime=timer()
# for i in range(loops):
#     print("%d. iteration: " % int(i+1))
#     #bigger sample random data
#     val=np.random.randint(1,1001,np.random.randint(4,51)).tolist()
#     wt=np.random.randint(1,1001,len(val)).tolist()
#     #random capacity
#     W=int(2*sum(wt)/np.random.randint(3,9))
#     KnapSackInstance=KnapSack(wt,val,W)
    
#     #get optimal value
#     opt=KnapSackInstance.dynApproachRec()['best_value']
#     #greedy max val
#     start = timer()
#     curr=KnapSackInstance.greedyMaxValue()['best_value']
#     timeResults['maxVal'][i]=timer()-start
#     distanceResults['maxVal'][i]=opt-curr

#     #greedy min distance
#     start = timer()
#     curr=KnapSackInstance.greedyMinWeight()['best_value']
#     timeResults['minWeight'][i]=timer()-start
#     distanceResults['minWeight'][i]=opt-curr
    
#     #greedy max ratio
#     start = timer()
#     curr=KnapSackInstance.greedyMaxRatio()['best_value']
#     timeResults['maxRatio'][i]=timer()-start
#     distanceResults['maxRatio'][i]=opt-curr
    
#     ###genetic algs with different setting
    
#     #default settings
#     start = timer()
#     curr=KnapSackInstance.geneticAlg()['best_value']
#     timeResults['geneticAlg - default'][i]=timer()-start
#     distanceResults['geneticAlg - default'][i]=opt-curr
    
#     #default settings, more iterations
#     start = timer()
#     curr=KnapSackInstance.geneticAlg(itCount=300)['best_value']
#     timeResults['geneticAlg - default extra iterations'][i]=timer()-start
#     distanceResults['geneticAlg - default extra iterations'][i]=opt-curr
    
#     #time-prefered
#     start = timer()
#     curr=KnapSackInstance.geneticAlg(popCount=30, itCount=50,crossRate=0.8,mutRate=0.1,prob=0.1)['best_value']
#     timeResults['geneticAlg - time'][i]=timer()-start
#     distanceResults['geneticAlg - time'][i]=opt-curr
    
#     #accuracy-prefered
#     start = timer()
#     curr=KnapSackInstance.geneticAlg(popCount=30, itCount=200,crossRate=0.6,mutRate=0.2,prob=0.2)['best_value']
#     timeResults['geneticAlg - accuracy'][i]=timer()-start
#     distanceResults['geneticAlg - accuracy'][i]=opt-curr
    
# print("%d iterations completed, total time of execution: %.2f s." % (loops,timer()-totalTime))#loop through the dictionary
# for key,value in timeResults.items(): 
#     #use reduce to calculate the avg
#     print(key,": elapsed time: %.4fs," % float(ft.reduce(lambda x, y: x + y, timeResults[key]) / len(timeResults[key])),
#           "distance from optimum: %d" % int(ft.reduce(lambda x, y: x + y, distanceResults[key]) / len(distanceResults[key])))



#best alg and heuristic comparison
loops=10
timeResults={"algorithm":np.empty(loops),"heuristics":np.empty(loops)}
accuracy=np.empty(loops)
for i in range(loops):
        print("%d. iteration: " % int(i+1))
        #bigger sample random data
        val=np.random.randint(1,101,np.random.randint(4,5)).tolist()
        wt=np.random.randint(1,101,len(val)).tolist()
        #random capacity
        W=int(2*sum(wt)/np.random.randint(3,9))
        KnapSackInstance=KnapSack(wt,val,W)
    
        start = timer()
        opt=KnapSackInstance.dynApproachRec()['best_value']
        timeResults['algorithm'][i]=timer()-start

        start = timer()
        curr=KnapSackInstance.greedyMaxRatio()['best_value']
        timeResults['heuristics'][i]=timer()-start
        accuracy[i]=opt-curr
for key,value in timeResults.items(): 
#use reduce to calculate the avg
    print(key,": elapsed time: %.4fs," % float(ft.reduce(lambda x, y: x + y, timeResults[key]) / len(timeResults[key])))
print("heuristics distance from optimum: %d" % int(ft.reduce(lambda x, y: x + y, accuracy) / len(accuracy)))
   