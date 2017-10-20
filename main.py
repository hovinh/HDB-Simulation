import numpy as np
import random 
import matplotlib.pyplot as plt

from scipy.optimize import linprog
from numpy.linalg import solve
from pulp import * 
from utility import *
from solver import *

'''GLOBAL PARAMETER'''

def set_global_parameter():
    global NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK, NUMB_TOTAL_FLATS, NUMB_ETHNICS, ACTUAL_RATIO, MAX_ITERATIONS
    NUMB_AGENTS = 50
    NUMB_BLOCKS = 5
    NUMB_FLATS_PER_BLOCK = [10, 10, 10, 10, 10]
    NUMB_TOTAL_FLATS = np.sum(NUMB_FLATS_PER_BLOCK)
    
    NUMB_ETHNICS = 3
    ACTUAL_RATIO = {'CHINESE': .77, 'MALAYS': .14, 'INDIANS': .08}
    MAX_ITERATIONS = 10000 # iterations of SIMPLEX ALGORITHM

    
'''MODELS'''

def unconstrained_model_random_utility():
    random_utility = RandomUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
    random_utility.generate()
    random_utility.plot_all()

    solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                             NUMB_FLATS_PER_BLOCK, random_utility)
    solver.calculate()
    
    print ('Status:', LpStatus[solver._prob.status])    
    print ('Optimal value:', value(solver._prob.objective))
    return solver

def calculate_optimal_constrained_model(utility_of_agents, ethnic_agents,
                                        ethnic_capacity_per_block):
    
    prob = LpProblem('HDB_constrained_random', LpMaximize)
    
    # Variables
    X = [str(i)+'_'+str(j) for i in range(NUMB_AGENTS) for j in range(NUMB_TOTAL_FLATS)]
    x_vars = LpVariable.dicts('x_vars', X, 0, 1, cat = 'Integer')
          
    # Objectives
    # coefficients of objective function
    utilities = np.ndarray.flatten(utility_of_agents)
    c = {}
    for i in range(NUMB_AGENTS * NUMB_TOTAL_FLATS):
        x_i = X[i]; utility = utilities[i]
        c[x_i] = utility

    prob += lpSum([c[i] * x_vars[i] for i in X])

    # Constraints
    # for each agent, there is at most one allocation
    A = []
    A_agent = np.zeros([NUMB_AGENTS, NUMB_AGENTS * NUMB_TOTAL_FLATS])
    for agent_index in range(NUMB_AGENTS):
        start_index = agent_index * NUMB_TOTAL_FLATS
        end_index = (agent_index + 1) * NUMB_TOTAL_FLATS
        A_agent[agent_index][start_index:end_index] = np.ones([1, NUMB_TOTAL_FLATS])
        A_temp = {}
        for i in range(NUMB_AGENTS * NUMB_TOTAL_FLATS):
            x_i = X[i]; indicator = A_agent[agent_index][i]
            A_temp[x_i] = indicator
        A.append(A_temp)
        
    # for each flat, there is at most one allocation
    A_flat = np.zeros([NUMB_TOTAL_FLATS, NUMB_AGENTS * NUMB_TOTAL_FLATS])
    for flat_index in range(NUMB_TOTAL_FLATS):
        flat_positions = [i for i in range(flat_index, NUMB_AGENTS * NUMB_TOTAL_FLATS,
                                             NUMB_TOTAL_FLATS)]
        A_flat[flat_index][flat_positions] = np.ones([1, NUMB_AGENTS])     
        A_temp = {}
        for i in range(NUMB_AGENTS * NUMB_TOTAL_FLATS):
            x_i = X[i]; indicator = A_flat[flat_index][i]
            A_temp[x_i] = indicator
        A.append(A_temp)
    
    # for each ethnicity, there is a limited capacity
    A_ethnic = np.zeros([NUMB_BLOCKS * NUMB_ETHNICS, NUMB_AGENTS * NUMB_TOTAL_FLATS])
    for block_index in range(NUMB_BLOCKS):
        start_index = block_index * NUMB_ETHNICS
        block_start = block_index * NUMB_FLATS_PER_BLOCK
        block_end = (block_index + 1) * NUMB_FLATS_PER_BLOCK

        # Chinese
        ethnic_positions = ethnic_agents == 'C'
        print (ethnic_positions[:])
        A_ethnic[start_index + 0][block_start:block_end] = ethnic_positions*1
        
        A_temp = {}
        for i in range(NUMB_AGENTS * NUMB_TOTAL_FLATS):
            x_i = X[i]; indicator = A_ethnic[start_index+0][i]
            A_temp[x_i] = indicator
        A.append(A_temp)

        # Malays
        ethnic_positions = ethnic_agents == 'M'
        A_ethnic[start_index + 1][block_start:block_end] = ethnic_positions*1
        
        A_temp = {}
        for i in range(NUMB_AGENTS * NUMB_TOTAL_FLATS):
            x_i = X[i]; indicator = A_ethnic[start_index+1][i]
            A_temp[x_i] = indicator
        A.append(A_temp)

        # Indians
        ethnic_positions = ethnic_agents == 'I'
        A_ethnic[start_index + 2][block_start:block_end] = ethnic_positions*1
            
        A_temp = {}
        for i in range(NUMB_AGENTS * NUMB_TOTAL_FLATS):
            x_i = X[i]; indicator = A_ethnic[start_index+2][i]
            A_temp[x_i] = indicator
        A.append(A_temp)

    for i in range(NUMB_AGENTS + NUMB_TOTAL_FLATS):
        prob += lpSum(A[i][j] * x_vars[j] for j in X) <= 1
    
    for i in range(NUMB_AGENTS + NUMB_TOTAL_FLATS, 
                   NUMB_AGENTS + NUMB_TOTAL_FLATS + NUMB_BLOCKS * NUMB_ETHNICS):
        block_index = i % (NUMB_AGENTS + NUMB_TOTAL_FLATS) 
        if (block_index == 0):
            prob += lpSum(A[i][j] * x_vars[j] for j in X) <= ethnic_capacity_per_block[block_index]['CHINESE']
        elif (block_index == 1):
            prob += lpSum(A[i][j] * x_vars[j] for j in X) <= ethnic_capacity_per_block[block_index]['MALAY']
        else:
            prob += lpSum(A[i][j] * x_vars[j] for j in X) <= ethnic_capacity_per_block[block_index]['INDIAN']

    GLPK().solve(prob)
    
    # Solution
    return res
    
def constrained_model_random_utility():
    
    # Assign randomly ethnic to agents according to actual proportion over population
    ethnic_capacity_per_block = [{'CHINESE': 0.847, 'MALAY': 0.154, 'INDIAN': 0.088}
                                 for i in range(NUMB_BLOCKS)]
    numb_chinese, numb_malays, numb_indians = 0, 0, 0
    for block_index in range(NUMB_BLOCKS):
        chinese = int (NUMB_FLATS_PER_BLOCK * ACTUAL_CHINESE_RATIO)
        malays = int (NUMB_FLATS_PER_BLOCK * ACTUAL_MALAYS_RATIO)
        indians = NUMB_FLATS_PER_BLOCK - chinese - malays
        numb_chinese += chinese; numb_malays += malays; numb_indians += indians
        
    ethnic_agents = np.full((1, numb_chinese), 'C')[0]
    ethnic_agents = np.concatenate((ethnic_agents, np.full((1, numb_malays), 'M')[0]),
                                   axis = 0)
    ethnic_agents = np.concatenate((ethnic_agents, np.full((1, numb_indians), 'I')[0]),
                                   axis = 0)
    ethnic_agents = np.random.permutation(ethnic_agents)
    
    utility_of_agents = generate_utility_randomly()
    optimal_value = calculate_optimal_constrained_model(utility_of_agents, 
                                                        ethnic_agents, ethnic_capacity_per_block)
    print ('Optimal value:', value(optimal_value.objective))
    return optimal_value    
    
if __name__ == '__main__':
    # turn stochastic process into deterministic, but turn off when experimenting
    np.random.seed(0) 
    random.seed(0)

    # set global parameter
    set_global_parameter()

    # list of simulation models    
    model_list = {'unconstrained_random': unconstrained_model_random_utility,
                  'constrained_random': constrained_model_random_utility}
    
    # execute the model
    optimal_value = model_list['unconstrained_random']()
    #optimal_value = model_list['constrained_random']()