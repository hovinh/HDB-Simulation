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
    NUMB_AGENTS = 2965
    NUMB_BLOCKS = 21
    NUMB_FLATS_PER_BLOCK = read_NUMB_FLATS_PER_BLOCK()
    NUMB_TOTAL_FLATS = np.sum(NUMB_FLATS_PER_BLOCK)
    
    NUMB_ETHNICS = 3
    ACTUAL_RATIO = {'CHINESE': .77, 'MALAYS': .14, 'INDIANS': .08}
    MAX_ITERATIONS = 600 # iterations of SIMPLEX ALGORITHM: 600 secs = 10 mins

def read_NUMB_FLATS_PER_BLOCK():
    numb_flats_per_block = []
    with open('actual_flats_per_block.txt', 'r') as file:
        for line in file:
            line = line.strip()
            numb_flats_per_block.append(int(line))
    
    return numb_flats_per_block
    
'''MODELS'''

def unconstrained_ip_random(should_plot = False):
    random_utility = RandomUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
    random_utility.generate()
    
    if (should_plot == True):
        random_utility.plot_all('Histogram of utility generated from Beta Distribution')
        random_utility.plot_per_block()

    solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                             NUMB_FLATS_PER_BLOCK, random_utility)
    solver.calculate(has_ethnicity=False)
    
    print ('Status:', LpStatus[solver._prob.status])    
    print ('Optimal value:', value(solver._prob.objective))
    return solver

def constrained_ip_random(should_plot = False):
    random_utility = RandomUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
    random_utility.add_ethnicity()
    random_utility.generate()
    
    if (should_plot == True):
        random_utility.plot_all('Histogram of utility generated from Beta Distribution')
        random_utility.plot_per_block()

    solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                             NUMB_FLATS_PER_BLOCK, random_utility)
    solver.add_ethnicity(random_utility, read_from_file=True, file_name='actual_ethnic.txt')
    solver.calculate(has_ethnicity = True)
  
    print ('Status:', LpStatus[solver._prob.status])    
    print ('Optimal value:', value(solver._prob.objective))
    return solver

def compare_model_ip_random(numb_iterations = 10):
    unconstrained_results = []
    constrained_results = []

    for iteration in range(numb_iterations):
        print ('Iteration:', iteration + 1)
        random_utility = RandomUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        random_utility.add_ethnicity()
        random_utility.generate()
        solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, random_utility)
        solver.calculate(has_ethnicity=False)
        unconstrained_results.append(value(solver._prob.objective))
        
        solver.add_ethnicity(random_utility, read_from_file=True, file_name='test_ethnic.txt')
        solver.calculate(has_ethnicity=True)
        constrained_results.append(value(solver._prob.objective))

    average_unconstrained_result = np.sum(unconstrained_results) / numb_iterations        
    average_constrained_result = np.sum(constrained_results) / numb_iterations

    print ('Unconstrained model:', average_unconstrained_result)
    print ('Constrained model:', average_constrained_result)
    print ('Ratio:', average_unconstrained_result / average_constrained_result)
 
def unconstrained_ip_location(should_plot = False):
    location_utility = LocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
    area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
    points_of_interest = location_utility.generate_points_of_interest(area)
    block_locations = location_utility.read_block_locations()
    location_utility.generate(points_of_interest, block_locations)
    
    if (should_plot == True):
        location_utility.plot_all('Histogram of utility generated from Block Location')
        location_utility.plot_per_block()

    solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                             NUMB_FLATS_PER_BLOCK, location_utility)
    solver.calculate(has_ethnicity=False)
    
    print ('Status:', LpStatus[solver._prob.status])    
    print ('Optimal value:', value(solver._prob.objective))
    
    return solver

def constrained_ip_location(should_plot = False):
    location_utility = LocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
    area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
    points_of_interest = location_utility.generate_points_of_interest(area)
    block_locations = location_utility.read_block_locations()
    location_utility.add_ethnicity()
    location_utility.generate(points_of_interest, block_locations)
    
    if (should_plot == True):
        location_utility.plot_all('Histogram of utility generated from Block Location')
        location_utility.plot_per_block()
    
    solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                             NUMB_FLATS_PER_BLOCK, location_utility)
    solver.add_ethnicity(location_utility, read_from_file=True, file_name='test_ethnic.txt')

    solver.calculate(has_ethnicity = True)
  
    print ('Status:', LpStatus[solver._prob.status])    
    print ('Optimal value:', value(solver._prob.objective))
    return solver
 
def compare_model_ip_location(numb_iterations = 10):
    unconstrained_results = []
    constrained_results = []

    for iteration in range(numb_iterations):
        print ('Iteration:', iteration + 1)
        location_utility = LocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
        points_of_interest = location_utility.generate_points_of_interest(area)
        block_locations = location_utility.read_block_locations()
        location_utility.add_ethnicity()
        location_utility.generate(points_of_interest, block_locations)
        
        
        solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, location_utility)
        solver.calculate(has_ethnicity=False)
        unconstrained_results.append(value(solver._prob.objective))
        
        solver.add_ethnicity(location_utility, read_from_file=True, file_name='test_ethnic.txt')
        solver.calculate(has_ethnicity=True)
        constrained_results.append(value(solver._prob.objective))


    average_unconstrained_result = np.sum(unconstrained_results) / numb_iterations        
    average_constrained_result = np.sum(constrained_results) / numb_iterations

    print ('Unconstrained model:', average_unconstrained_result)
    print ('Constrained model:', average_constrained_result)
    print ('Ratio:', average_unconstrained_result / average_constrained_result)
     
def unconstrained_lottery_random(should_plot = False):
    random_utility = RandomUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
    random_utility.generate()
    
    if (should_plot == True):
        random_utility.plot_all('Histogram of utility generated from Beta Distribution')
        random_utility.plot_per_block()

    solver = LotterySolver(NUMB_AGENTS, NUMB_BLOCKS,
                                             NUMB_FLATS_PER_BLOCK, random_utility)
    x_vars, optimal_value = solver.calculate(has_ethnicity=False)

    print ('Optimal value:', optimal_value)
    print ('Number of assigned agents', np.sum(x_vars)) 
    
    return optimal_value

def constrained_lottery_random(should_plot = False):
    random_utility = RandomUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
    random_utility.add_ethnicity()
    random_utility.generate()
    
    if (should_plot == True):
        random_utility.plot_all('Histogram of utility generated from Beta Distribution')
        random_utility.plot_per_block()

    solver = LotterySolver(NUMB_AGENTS, NUMB_BLOCKS,
                                             NUMB_FLATS_PER_BLOCK, random_utility)
    solver.add_ethnicity(random_utility, read_from_file=True, file_name='test_ethnic.txt')
    x_vars, optimal_value = solver.calculate(has_ethnicity=True)

    print ('Optimal value:', optimal_value)
    print ('Number of assigned agents', np.sum(x_vars))
    
    return optimal_value

def compare_model_lottery_random(numb_iterations = 10):
    unconstrained_results = []
    constrained_results = []

    for iteration in range(numb_iterations):
        print ('Iteration:', iteration + 1)
        random_utility = RandomUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        random_utility.add_ethnicity()
        random_utility.generate()
        solver = LotterySolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, random_utility)
        x_vars, optimal_value = solver.calculate(has_ethnicity=False)
        unconstrained_results.append(optimal_value)
        
        solver.add_ethnicity(random_utility, read_from_file=True, file_name='test_ethnic.txt')
        x_vars, optimal_value = solver.calculate(has_ethnicity=True)
        constrained_results.append(optimal_value)

    average_unconstrained_result = np.sum(unconstrained_results) / numb_iterations        
    average_constrained_result = np.sum(constrained_results) / numb_iterations

    print ('Unconstrained model:', average_unconstrained_result)
    print ('Constrained model:', average_constrained_result)
    print ('Ratio:', average_unconstrained_result / average_constrained_result)
    
def unconstrained_lottery_location(should_plot = False):
    location_utility = LocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
    area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
    points_of_interest = location_utility.generate_points_of_interest(area)
    block_locations = location_utility.read_block_locations()
    location_utility.generate(points_of_interest, block_locations)
    
    if (should_plot == True):
        location_utility.plot_all('Histogram of utility generated from Block Location')
        location_utility.plot_per_block()

    solver = LotterySolver(NUMB_AGENTS, NUMB_BLOCKS,
                                             NUMB_FLATS_PER_BLOCK, location_utility)
    x_vars, optimal_value = solver.calculate(has_ethnicity=False)

    print ('Optimal value:', optimal_value)
    print ('Number of assigned agents', np.sum(x_vars))
    
    return optimal_value

def constrained_lottery_location(should_plot = False):
    location_utility = LocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
    area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
    points_of_interest = location_utility.generate_points_of_interest(area)
    block_locations = location_utility.read_block_locations()
    location_utility.add_ethnicity()
    location_utility.generate(points_of_interest, block_locations)
    
    if (should_plot == True):
        location_utility.plot_all('Histogram of utility generated from Block Location')
        location_utility.plot_per_block()

    solver = LotterySolver(NUMB_AGENTS, NUMB_BLOCKS,
                                             NUMB_FLATS_PER_BLOCK, location_utility)
    solver.add_ethnicity(location_utility, read_from_file=True, file_name='test_ethnic.txt')
    x_vars, optimal_value = solver.calculate(has_ethnicity=True)

    print ('Optimal value:', optimal_value)
    print ('Number of assigned agents', np.sum(x_vars))
  
def compare_model_lottery_location(numb_iterations = 10):
    unconstrained_results = []
    constrained_results = []

    for iteration in range(numb_iterations):
        print ('Iteration:', iteration + 1)
        location_utility = LocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
        points_of_interest = location_utility.generate_points_of_interest(area)
        block_locations = location_utility.read_block_locations()
        location_utility.add_ethnicity()
        location_utility.generate(points_of_interest, block_locations)
        
        solver = LotterySolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, location_utility)
        x_vars, optimal_value = solver.calculate(has_ethnicity=False)
        unconstrained_results.append(optimal_value)
        
        solver.add_ethnicity(location_utility, read_from_file=True, file_name='test_ethnic.txt')
        x_vars, optimal_value = solver.calculate(has_ethnicity=True)
        constrained_results.append(optimal_value)

    average_unconstrained_result = np.sum(unconstrained_results) / numb_iterations        
    average_constrained_result = np.sum(constrained_results) / numb_iterations

    print ('Unconstrained model:', average_unconstrained_result)
    print ('Constrained model:', average_constrained_result)
    print ('Ratio:', average_unconstrained_result / average_constrained_result)
    
if __name__ == '__main__':
    # turn stochastic process into deterministic, but turn off when experimenting
    np.random.seed(0) 
    random.seed(0)

    # set global parameter
    set_global_parameter()

    # list of simulation models    
    model_list = {'unconstrained_ipsolver_random': unconstrained_ip_random,
                  'constrained_ipsolver_random': constrained_ip_random,
                  'unconstrained_ipsolver_location': unconstrained_ip_location,
                  'constrained_ipsolver_location': constrained_ip_location,
                  'unconstrained_lotterysolver_random': unconstrained_lottery_random,
                  'constrained_lotterysolver_random': constrained_lottery_random,
                  'unconstrained_lotterysolver_location': unconstrained_lottery_location,
                  'constrained_lotterysolver_location': constrained_lottery_location}
    
    # execute the model
    optimal_value = model_list['unconstrained_ipsolver_random']()
    optimal_value = model_list['constrained_ipsolver_random']()
    #compare_model_ip_random(2)

    #optimal_value = model_list['unconstrained_ipsolver_location']()
    #optimal_value = model_list['constrained_ipsolver_location']()
    #compare_model_ip_location(2)
    
    #optimal_value = model_list['unconstrained_lotterysolver_random']()
    #optimal_value = model_list['constrained_lotterysolver_random']()
    #compare_model_lottery_random()
    
    #optimal_value = model_list['unconstrained_lotterysolver_location']()
    #optimal_value = model_list['constrained_lotterysolver_location']()
    #compare_model_lottery_location()