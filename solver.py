'''This is classes for Linear Programming Solver'''
import numpy as np
import abc
from pulp import *
from utility import *
'''
Abstract class Utility
'''
class Solver(abc.ABC):

    @abc.abstractmethod
    def __init__(self, numb_agents, numb_blocks, numb_flats_per_block, utility):
        self._solution = None
    
    @abc.abstractmethod
    def calculate(self):
        pass
    
    
    
'''
IntegerProgrammingSolver class
'''
class IntegerProgrammingSolver(Solver):
    
    def __init__(self, numb_agents, numb_blocks, numb_flats_per_block, utility):
        self._numb_agents = numb_agents
        self._numb_blocks = numb_blocks
        self._numb_flats_per_block = numb_flats_per_block
        self._numb_total_flats = np.sum(numb_flats_per_block)
        self._utility = utility._utility
        
    def calculate(self, has_ethnicity = False):
    
        prob = LpProblem('IntegerProgramming', LpMaximize)
        
        # Variables
        X = [str(i)+'_'+str(j) for i in range(self._numb_agents) for j in range(self._numb_total_flats)]
        x_vars = LpVariable.dicts('x_vars', X, 0, 1, cat = 'Integer')
              
        # Objectives
        # coefficients of objective function
        utilities = np.ndarray.flatten(self._utility)
        c = {}
        for i in range(self._numb_agents * self._numb_total_flats):
            x_i = X[i]; utility = utilities[i]
            c[x_i] = utility
    
        prob += lpSum([c[i] * x_vars[i] for i in X])
    
        # Constraints
        # for each agent, there is at most one allocation
        A = []
        A_agent = np.zeros([self._numb_agents, self._numb_agents * self._numb_total_flats])
        for agent_index in range(self._numb_agents):
            start_index = agent_index * self._numb_total_flats
            end_index = (agent_index + 1) * self._numb_total_flats
            A_agent[agent_index][start_index:end_index] = np.ones([1, self._numb_total_flats])
            A.append(self.convert_A_to_dict(X, A_agent[agent_index]))
            
        # for each flat, there is at most one allocation
        A_flat = np.zeros([self._numb_total_flats, self._numb_agents * self._numb_total_flats])
        for flat_index in range(self._numb_total_flats):
            flat_positions = [i for i in range(flat_index, self._numb_agents * self._numb_total_flats,
                                                 self._numb_total_flats)]
            A_flat[flat_index][flat_positions] = np.ones([1, self._numb_agents])     
            A.append(self.convert_A_to_dict(X, A_flat[flat_index]))
                        
        for i in range(self._numb_agents + self._numb_total_flats):
            prob += lpSum(A[i][j] * x_vars[j] for j in X) <= 1

        # for each block, there is a limited capacity for each ethnicity
        if (has_ethnicity == True):
            numb_ethnicities = len(self._ethnic_capacity_per_block[0])
            
            for block_index in range(self._numb_blocks):
                
                for ethnicity_index in range(numb_ethnicities):
                    
                    row_index = block_index * self._numb_blocks + ethnicity_index
                    '''TODO HERE'''
                    
                    
            for i in range(self._numb_agents + self._numb_total_flats,
                           self._numb_agents + self._numb_total_flats 
                           + self._numb_blocks * numb_ethnicities):
                ethnic_index = i % (self._numb_agents + self._numb_total_flats) 
                block_index = (i - (self._numb_agents + self._numb_total_flats)) / numb_ethnicities
                prob += lpSum(A[i][j] * x_vars[j] for j in X) <= self._ethnic_capacity_per_block[block_index][ethnic_index]
                
        GLPK().solve(prob)

        self._prob = prob
        
        # Solution
        return prob
    
    def convert_A_to_dict(self, X, A_sub):
        A_temp = {}
        for i in range(self._numb_agents * self._numb_total_flats):
            x_i = X[i]; indicator = A_sub[i]
            A_temp[x_i] = indicator
        return A_temp
    
    def add_ethnicity(self, utility, read_from_file = False, ethnic_capacity_per_block = None):
        self._ethnic_capacity_per_block = None
        if (read_from_file == False):
            self._ethnic_capacity_per_block = ethnic_capacity_per_block
        else:
            self._ethnic_capacity_per_block = self.read_ethnic_capacity_per_block()
        
        self._ethnic_agents = self.read_utility_ethnic_agents(utility._ethnic_agents)

    def read_ethnic_capacity_per_block(self):
        ethnic_capacity_per_block = []
        with open('ethnic_capacity_per_block.txt', 'r') as file:
            for line in file:
                line = line.strip()
                ethnic_capacity_per_block.append([int(i) for i in line.split(',')])
        return ethnic_capacity_per_block
        
    def read_utility_ethnic_agents(self, utility_ethnic_agents):
        ethnic_agents = np.full((1, self._numb_total_flats), 'Q')
        for ethnic_agent in utility_ethnic_agents:
            agent_duplications = np.full((1, self._numb_total_flats), ethnic_agent)
            ethnic_agents = np.concatenate((ethnic_agents, agent_duplications), axis = 0)
        ethnic_agents = np.ndarray.flatten(ethnic_agents[1:])
        return np.array(ethnic_agents)
        
def test1():
    np.random.seed(0)
    numb_agents = 50
    numb_blocks = 5
    numb_flats_per_block = [10, 10, 10, 10, 10]
    
    utility = RandomUtility(numb_agents, numb_blocks, numb_flats_per_block)
    utility.generate()

    solver = IntegerProgrammingSolver(numb_agents, numb_blocks, numb_flats_per_block,
                                 utility)
    solver.calculate()

    print ('Status:', LpStatus[solver._prob.status])    
    print ('Optimal value:', value(solver._prob.objective))

def test2():
    np.random.seed(0)
    numb_agents = 50
    numb_blocks = 5
    numb_flats_per_block = [10, 10, 10, 10, 10]
    
    utility = RandomUtility(numb_agents, numb_blocks, numb_flats_per_block)
    utility.generate()

    solver = IntegerProgrammingSolver(numb_agents, numb_blocks, numb_flats_per_block,
                                 utility)
    try:
        print ('Status:', LpStatus[solver._prob.status])
        print ('Optimal value:', value(solver._prob.objective))
    except AttributeError:
        print ('Could not access to solution prior to generating step')
        
def test3():
    np.random.seed(0)
    numb_agents = 10
    numb_blocks = 5
    numb_flats_per_block = [10, 10, 10, 10, 10]
    
    utility = RandomUtility(numb_agents, numb_blocks, numb_flats_per_block)
    utility.generate()

    solver = IntegerProgrammingSolver(numb_agents, numb_blocks, numb_flats_per_block,
                                 utility)
    solver.calculate()
    
    print ('Status:', LpStatus[solver._prob.status])    
    print ('Optimal value:', value(solver._prob.objective))

def test4():
    np.random.seed(0)
    numb_agents = 50
    numb_blocks = 5
    numb_flats_per_block = [10, 10, 10, 10, 10]
    
    utility = RandomUtility(numb_agents, numb_blocks, numb_flats_per_block)
    utility.add_ethnicity()
    utility.generate()

    solver = IntegerProgrammingSolver(numb_agents, numb_blocks, numb_flats_per_block,
                                 utility)
    ethnic_capacity_ratio_per_block = [{'CHINESE': 0.847, 'MALAY': 0.154, 'INDIAN': 0.088}
                                 for i in range(numb_blocks)]
    ethnic_capacity_per_block = []
    for i in range(numb_blocks):
        chinese_capacity = int(numb_flats_per_block[i] * ethnic_capacity_ratio_per_block[i]['CHINESE'])
        malay_capacity = int(numb_flats_per_block[i] * ethnic_capacity_ratio_per_block[i]['MALAY'])
        indian_capacity = int(numb_flats_per_block[i] * ethnic_capacity_ratio_per_block[i]['INDIAN'])
        ethnic_capacity_per_block.append([chinese_capacity, malay_capacity, indian_capacity])
        
    solver.add_ethnicity(utility, False, ethnic_capacity_per_block)
    print (solver._ethnic_capacity_per_block)
    print (solver._ethnic_agents.shape)
    solver.calculate()
    
    print ('Status:', LpStatus[solver._prob.status])    
    print ('Optimal value:', value(solver._prob.objective))

def test5():
    np.random.seed(0)
    numb_agents = 50
    numb_blocks = 5
    numb_flats_per_block = [10, 10, 10, 10, 10]
    
    utility = RandomUtility(numb_agents, numb_blocks, numb_flats_per_block)
    utility.add_ethnicity()
    utility.generate()

    solver = IntegerProgrammingSolver(numb_agents, numb_blocks, numb_flats_per_block,
                                 utility)
    
    solver.add_ethnicity(utility, True)
    print (solver._ethnic_capacity_per_block)
    print (solver._ethnic_agents.shape)
    
if __name__ == '__main__':
    ''' Test suite for IntegerProgrammingSolver '''
    #test1()
    #test2()
    #test3() # costly, check for difference between # agents and # flats
    test4()
    test5()   