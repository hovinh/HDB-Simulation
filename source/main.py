import numpy as np
import random 
import matplotlib.pyplot as plt
import time

from numpy.linalg import solve
from pulp import * 
from utility import *
from solver import *

'''GLOBAL PARAMETER'''

def set_global_parameter_test():
    global NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK, NUMB_TOTAL_FLATS, NUMB_ETHNICS, ACTUAL_RATIO, MAX_ITERATIONS, ETHNIC_FILE, BLOCK_LOCATION_FILE
    NUMB_AGENTS = 50
    NUMB_BLOCKS = 5
    NUMB_FLATS_PER_BLOCK = read_NUMB_FLATS_PER_BLOCK('test_flats_per_block.txt')
    NUMB_TOTAL_FLATS = np.sum(NUMB_FLATS_PER_BLOCK)
    
    BLOCK_LOCATION_FILE = 'test_location.txt'
    ETHNIC_FILE = 'test_ethnic.txt'
    NUMB_ETHNICS = 3
    ACTUAL_RATIO = {'CHINESE': .77, 'MALAYS': .14, 'INDIANS': .08}
    MAX_ITERATIONS = 600 # iterations of SIMPLEX ALGORITHM: 600 secs = 10 mins

def set_global_parameter_actual():
    global NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK, NUMB_TOTAL_FLATS, NUMB_ETHNICS, ACTUAL_RATIO, MAX_ITERATIONS, ETHNIC_FILE, BLOCK_LOCATION_FILE
    NUMB_AGENTS = 123
    NUMB_BLOCKS = 2
    NUMB_FLATS_PER_BLOCK = read_NUMB_FLATS_PER_BLOCK('actual_flats_per_block.txt')
    NUMB_TOTAL_FLATS = np.sum(NUMB_FLATS_PER_BLOCK)
    
    BLOCK_LOCATION_FILE = 'actual_location.txt'
    ETHNIC_FILE = 'actual_ethnic.txt'
    NUMB_ETHNICS = 3
    ACTUAL_RATIO = {'CHINESE': .77, 'MALAYS': .14, 'INDIANS': .08}
    MAX_ITERATIONS = 600 # iterations of SIMPLEX ALGORITHM: 600 secs = 10 mins

    
def read_NUMB_FLATS_PER_BLOCK(file_name = 'test_flats_per_block.txt'):
    numb_flats_per_block = []
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()
            numb_flats_per_block.append(int(line))
    
    return numb_flats_per_block
    
def calculate_expected_utility_wrt_ethnic(x_vars, ethnic_agents, utilities):
    numb_agents_ethnic = {'C':0., 'M':0., 'I':0.}
    utilities_ethnic = {'C':0., 'M':0., 'I':0.}

    for agent_index in range(NUMB_AGENTS):
        ethnic = ethnic_agents[agent_index]
        numb_agents_ethnic[ethnic] += 1
        x = x_vars[agent_index*NUMB_TOTAL_FLATS : (agent_index+1)*NUMB_TOTAL_FLATS]
        utility = utilities[agent_index]
        utilities_ethnic[ethnic] += np.sum(x*utility)
    expected = {}
    for ethnic in ['C', 'M', 'I']:
        expected[ethnic] = utilities_ethnic[ethnic] / numb_agents_ethnic[ethnic]

    return expected
    
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
    x_vars = [v.varValue for v in solver._prob.variables()]
    random_utility.add_ethnicity()
    expected = calculate_expected_utility_wrt_ethnic(x_vars, random_utility._ethnic_agents, 
                                                     random_utility._utility)
    print (expected)
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
    solver.add_ethnicity(random_utility, read_from_file=True, file_name=ETHNIC_FILE)
    solver.calculate(has_ethnicity = True)
    
    x_vars = [v.varValue for v in solver._prob.variables()]
    random_utility.add_ethnicity()
    expected = calculate_expected_utility_wrt_ethnic(x_vars, random_utility._ethnic_agents, 
                                                     random_utility._utility)
    print (expected)
    print ('Status:', LpStatus[solver._prob.status])    
    print ('Optimal value:', value(solver._prob.objective))
    return solver

def compare_model_ip_random(numb_iterations = 10):
    unconstrained_results = []
    constrained_results = []
    unconstrained_expected = []
    constrained_expected = []

    for iteration in range(numb_iterations):
        print ('Iteration:', iteration + 1)
        np.random.seed(iteration)
        random.seed(iteration)
        random_utility = RandomUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        random_utility.add_ethnicity()
        random_utility.generate()
        solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, random_utility)
        solver.calculate(has_ethnicity=False)
        unconstrained_results.append(value(solver._prob.objective))
        x_vars = [v.varValue for v in solver._prob.variables()]
        unconstrained_expected.append(calculate_expected_utility_wrt_ethnic(x_vars, random_utility._ethnic_agents,
                                                                            random_utility._utility))
        
        
        np.random.seed(iteration)
        random.seed(iteration)
        random_utility = RandomUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        random_utility.add_ethnicity()
        random_utility.generate()
        solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, random_utility)
        solver.add_ethnicity(random_utility, read_from_file=True, file_name=ETHNIC_FILE)
        solver.calculate(has_ethnicity=True)
        constrained_results.append(value(solver._prob.objective))
        x_vars = [v.varValue for v in solver._prob.variables()]
        constrained_expected.append(calculate_expected_utility_wrt_ethnic(x_vars, random_utility._ethnic_agents,
                                                                          random_utility._utility))

    average_unconstrained_result = np.sum(unconstrained_results) / numb_iterations        
    average_constrained_result = np.sum(constrained_results) / numb_iterations

    print ('Unconstrained model')
    print ('Average optimal: ', average_unconstrained_result)
    print (unconstrained_results)
    print (unconstrained_expected)
    print ('Constrained model')
    print ('Average optimal: ', average_constrained_result)
    print (constrained_results)
    print (constrained_expected)
    print ('Ratio:', average_unconstrained_result / average_constrained_result)
 
def unconstrained_ip_location(should_plot = False):
    location_utility = EthnicalLocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
    area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
    points_of_interest = location_utility.generate_ethnical_points_of_interest(area)
    block_locations = location_utility.read_block_locations(BLOCK_LOCATION_FILE)
    location_utility.add_ethnicity()
    location_utility.generate(points_of_interest, block_locations, set_variance=False)
    
    if (should_plot == True):
        location_utility.plot_all('Histogram of utility generated from Ethnical Points of Interest')
        location_utility.plot_per_block()

    solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                             NUMB_FLATS_PER_BLOCK, location_utility)
    solver.calculate(has_ethnicity=False)
    
    print ('Status:', LpStatus[solver._prob.status])    
    print ('Optimal value:', value(solver._prob.objective))
    
    return solver

def constrained_ip_location(should_plot = False):
    location_utility = EthnicalLocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
    area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
    points_of_interest = location_utility.generate_ethnical_points_of_interest(area)
    block_locations = location_utility.read_block_locations(BLOCK_LOCATION_FILE)
    location_utility.add_ethnicity()
    location_utility.generate(points_of_interest, block_locations, set_variance=False)
    
    if (should_plot == True):
        location_utility.plot_all('Histogram of utility generated from Ethical Points of Interest')
        location_utility.plot_per_block()
    
    solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                             NUMB_FLATS_PER_BLOCK, location_utility)
    solver.add_ethnicity(location_utility, read_from_file=True, file_name=ETHNIC_FILE)

    solver.calculate(has_ethnicity = True)
  
    print ('Status:', LpStatus[solver._prob.status])    
    print ('Optimal value:', value(solver._prob.objective))
    return solver
 
def compare_model_ip_location(numb_iterations = 10):
    unconstrained_results = []
    constrained_results = []
    unconstrained_expected = []
    constrained_expected = []

    for iteration in range(numb_iterations):
        print ('Iteration:', iteration + 1)
        np.random.seed(iteration)
        random.seed(iteration)
        location_utility = EthnicalLocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
        points_of_interest = location_utility.generate_ethnical_points_of_interest(area)
        block_locations = location_utility.read_block_locations(BLOCK_LOCATION_FILE)
        location_utility.add_ethnicity()
        location_utility.generate(points_of_interest, block_locations, set_variance=False)
        solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, location_utility)
        solver.calculate(has_ethnicity=False)
        unconstrained_results.append(value(solver._prob.objective))
        x_vars = [v.varValue for v in solver._prob.variables()]
        unconstrained_expected.append(calculate_expected_utility_wrt_ethnic(x_vars, location_utility._ethnic_agents,
                                                                            location_utility._utility))

        
        
        np.random.seed(iteration)
        random.seed(iteration)
        location_utility = EthnicalLocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
        points_of_interest = location_utility.generate_ethnical_points_of_interest(area)
        block_locations = location_utility.read_block_locations(BLOCK_LOCATION_FILE)
        location_utility.add_ethnicity()
        location_utility.generate(points_of_interest, block_locations, set_variance=False)
        solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, location_utility)
        solver.add_ethnicity(location_utility, read_from_file=True, file_name=ETHNIC_FILE)
        solver.calculate(has_ethnicity=True)
        constrained_results.append(value(solver._prob.objective))
        x_vars = [v.varValue for v in solver._prob.variables()]
        constrained_expected.append(calculate_expected_utility_wrt_ethnic(x_vars, location_utility._ethnic_agents,
                                                                            location_utility._utility))


    average_unconstrained_result = np.sum(unconstrained_results) / numb_iterations        
    average_constrained_result = np.sum(constrained_results) / numb_iterations

    print ('Unconstrained model')
    print ('Average optimal: ', average_unconstrained_result)
    print (unconstrained_results)
    print (unconstrained_expected)
    print ('Constrained model')
    print ('Average optimal: ', average_constrained_result)
    print (constrained_results)
    print (constrained_expected)
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
    random_utility.add_ethnicity()
    expected = calculate_expected_utility_wrt_ethnic(x_vars, random_utility._ethnic_agents, 
                                                     random_utility._utility)
    print ('Optimal value:', optimal_value)
    print ('Number of assigned agents', np.sum(x_vars)) 
    print (expected)   
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
    solver.add_ethnicity(random_utility, read_from_file=True, file_name=ETHNIC_FILE)
    x_vars, optimal_value = solver.calculate(has_ethnicity=True)
    expected = calculate_expected_utility_wrt_ethnic(x_vars, random_utility._ethnic_agents, 
                                                     random_utility._utility)
   
    print ('Optimal value:', optimal_value)
    print ('Number of assigned agents', np.sum(x_vars))
    print (expected)   
    return optimal_value

def compare_model_lottery_random(numb_iterations = 10):
    unconstrained_results = []
    constrained_results = []
    unconstrained_expected = []
    constrained_expected = []

    for iteration in range(numb_iterations):
        print ('Iteration:', iteration + 1)
        np.random.seed(iteration)
        random.seed(iteration)
        random_utility = RandomUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        random_utility.add_ethnicity()
        random_utility.generate()
        solver = LotterySolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, random_utility)
        x_vars, optimal_value = solver.calculate(has_ethnicity=False)
        unconstrained_results.append(optimal_value)
        unconstrained_expected.append(calculate_expected_utility_wrt_ethnic(x_vars, random_utility._ethnic_agents,
                                                                            random_utility._utility))
        
        np.random.seed(iteration)
        random.seed(iteration)
        random_utility = RandomUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        random_utility.add_ethnicity()
        random_utility.generate()
        solver = LotterySolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, random_utility)
        solver.add_ethnicity(random_utility, read_from_file=True, file_name=ETHNIC_FILE)
        x_vars, optimal_value = solver.calculate(has_ethnicity=True)
        constrained_results.append(optimal_value)
        constrained_expected.append(calculate_expected_utility_wrt_ethnic(x_vars, random_utility._ethnic_agents,
                                                                          random_utility._utility))

    average_unconstrained_result = np.sum(unconstrained_results) / numb_iterations        
    average_constrained_result = np.sum(constrained_results) / numb_iterations

    print ('Unconstrained model')
    print ('Average optimal: ', average_unconstrained_result)
    print (unconstrained_results)
    print (unconstrained_expected)
    print ('Constrained model')
    print ('Average optimal: ', average_constrained_result)
    print (constrained_results)
    print (constrained_expected)
    print ('Ratio:', average_unconstrained_result / average_constrained_result)    
    
def unconstrained_lottery_location(should_plot = False):
    location_utility = EthnicalLocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
    area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
    points_of_interest = location_utility.generate_ethnical_points_of_interest(area)
    block_locations = location_utility.read_block_locations(BLOCK_LOCATION_FILE)
    location_utility.add_ethnicity()
    location_utility.generate(points_of_interest, block_locations, set_variance=False)
    
    if (should_plot == True):
        location_utility.plot_all('Histogram of utility generated from Block Location')
        location_utility.plot_per_block()

    solver = LotterySolver(NUMB_AGENTS, NUMB_BLOCKS,
                                             NUMB_FLATS_PER_BLOCK, location_utility)
    x_vars, optimal_value = solver.calculate(has_ethnicity=False)
    expected = calculate_expected_utility_wrt_ethnic(x_vars, location_utility._ethnic_agents,
                                                     location_utility._utility)
    print ('Optimal value:', optimal_value)
    print ('Number of assigned agents', np.sum(x_vars))
    print (expected)
    return optimal_value

def constrained_lottery_location(should_plot = False):
    location_utility = EthnicalLocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
    area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
    points_of_interest = location_utility.generate_ethnical_points_of_interest(area)
    block_locations = location_utility.read_block_locations(BLOCK_LOCATION_FILE)
    location_utility.add_ethnicity()
    location_utility.generate(points_of_interest, block_locations, set_variance=False)
    
    if (should_plot == True):
        location_utility.plot_all('Histogram of utility generated from Block Location')
        location_utility.plot_per_block()

    solver = LotterySolver(NUMB_AGENTS, NUMB_BLOCKS,
                                             NUMB_FLATS_PER_BLOCK, location_utility)
    solver.add_ethnicity(location_utility, read_from_file=True, file_name=ETHNIC_FILE)
    x_vars, optimal_value = solver.calculate(has_ethnicity=True)
    expected = calculate_expected_utility_wrt_ethnic(x_vars, location_utility._ethnic_agents,
                                                     location_utility._utility)

    print ('Optimal value:', optimal_value)
    print ('Number of assigned agents', np.sum(x_vars))
    print (expected)
    return optimal_value

def compare_model_lottery_location(numb_iterations = 10):
    unconstrained_results = []
    constrained_results = []
    unconstrained_expected = []
    constrained_expected = []

    for iteration in range(numb_iterations):
        print ('Iteration:', iteration + 1)
        np.random.seed(iteration)
        random.seed(iteration)
        location_utility = EthnicalLocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
        points_of_interest = location_utility.generate_ethnical_points_of_interest(area)
        block_locations = location_utility.read_block_locations(BLOCK_LOCATION_FILE)
        location_utility.add_ethnicity()
        location_utility.generate(points_of_interest, block_locations, set_variance=False)
        
        solver = LotterySolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, location_utility)
        x_vars, optimal_value = solver.calculate(has_ethnicity=False)
        unconstrained_results.append(optimal_value)
        unconstrained_expected.append(calculate_expected_utility_wrt_ethnic(x_vars, location_utility._ethnic_agents,
                                                                            location_utility._utility))
        
        
        
        np.random.seed(iteration)
        random.seed(iteration)
        location_utility = EthnicalLocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
        points_of_interest = location_utility.generate_ethnical_points_of_interest(area)
        block_locations = location_utility.read_block_locations(BLOCK_LOCATION_FILE)
        location_utility.add_ethnicity()
        location_utility.generate(points_of_interest, block_locations, set_variance=False)
        
        solver = LotterySolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, location_utility)
        solver.add_ethnicity(location_utility, read_from_file=True, file_name=ETHNIC_FILE)
        x_vars, optimal_value = solver.calculate(has_ethnicity=True)
        constrained_results.append(optimal_value)
        constrained_expected.append(calculate_expected_utility_wrt_ethnic(x_vars, location_utility._ethnic_agents,
                                                                          location_utility._utility))

    average_unconstrained_result = np.sum(unconstrained_results) / numb_iterations        
    average_constrained_result = np.sum(constrained_results) / numb_iterations

    print ('Unconstrained model')
    print ('Average optimal: ', average_unconstrained_result)
    print (unconstrained_results)
    print (unconstrained_expected)
    print ('Constrained model')
    print ('Average optimal: ', average_constrained_result)
    print (constrained_results)
    print (constrained_expected)
    print ('Ratio:', average_unconstrained_result / average_constrained_result)

def compare_model_unconstrained_random(numb_iterations = 10):  
    integer_results = []
    lottery_results = []
    integer_expected = []
    lottery_expected = []

    for iteration in range(numb_iterations):
        print ('Iteration:', iteration + 1)
        np.random.seed(iteration)
        random.seed(iteration)
        random_utility = RandomUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        random_utility.add_ethnicity()
        random_utility.generate()
        solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, random_utility)
        solver.calculate(has_ethnicity=False)
        integer_results.append(value(solver._prob.objective))
        x_vars = [v.varValue for v in solver._prob.variables()]
        integer_expected.append(calculate_expected_utility_wrt_ethnic(x_vars, random_utility._ethnic_agents,
                                                                      random_utility._utility))
  
        np.random.seed(iteration)
        random.seed(iteration)
        random_utility = RandomUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        random_utility.add_ethnicity()
        random_utility.generate()
        solver = LotterySolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, random_utility)
        x_vars, optimal_value = solver.calculate(has_ethnicity=False)
        lottery_results.append(optimal_value)
        lottery_expected.append(calculate_expected_utility_wrt_ethnic(x_vars, random_utility._ethnic_agents,
                                                                      random_utility._utility))
        
                                                                      
    average_integer_result = np.sum(integer_results) / numb_iterations
    average_lottery_result = np.sum(lottery_results) / numb_iterations

    print ('Integer model')
    print ('Average optimal: ', average_integer_result)
    print (integer_results)
    print (integer_expected)
    print ('Lottery model')
    print ('Average optimal: ', average_lottery_result)
    print (lottery_results)
    print (lottery_expected)

def compare_model_constrained_random(numb_iterations = 10):
    integer_results = []
    lottery_results = []
    integer_expected = []
    lottery_expected = []

    for iteration in range(numb_iterations):
        print ('Iteration:', iteration + 1)
        np.random.seed(iteration)
        random.seed(iteration)
        random_utility = RandomUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        random_utility.add_ethnicity()
        random_utility.generate()
        solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, random_utility)
        solver.add_ethnicity(random_utility, read_from_file=True, file_name=ETHNIC_FILE)
        solver.calculate(has_ethnicity=True)
        integer_results.append(value(solver._prob.objective))
        x_vars = [v.varValue for v in solver._prob.variables()]
        integer_expected.append(calculate_expected_utility_wrt_ethnic(x_vars, random_utility._ethnic_agents,
                                                                      random_utility._utility))
  
        np.random.seed(iteration)
        random.seed(iteration)
        random_utility = RandomUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        random_utility.add_ethnicity()
        random_utility.generate()
        solver = LotterySolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, random_utility)
        solver.add_ethnicity(random_utility, read_from_file=True, file_name=ETHNIC_FILE)
        x_vars, optimal_value = solver.calculate(has_ethnicity=True)
        lottery_results.append(optimal_value)
        lottery_expected.append(calculate_expected_utility_wrt_ethnic(x_vars, random_utility._ethnic_agents,
                                                                      random_utility._utility))
        
                                                                      
    average_integer_result = np.sum(integer_results) / numb_iterations
    average_lottery_result = np.sum(lottery_results) / numb_iterations

    print ('Integer model')
    print ('Average optimal: ', average_integer_result)
    print (integer_results)
    print (integer_expected)
    print ('Lottery model')
    print ('Average optimal: ', average_lottery_result)
    print (lottery_results)
    print (lottery_expected)

def compare_model_unconstrained_location(numb_iterations = 10):
    integer_results = []
    lottery_results = []
    integer_expected = []
    lottery_expected = []

    for iteration in range(numb_iterations):
        print ('Iteration:', iteration + 1)
        np.random.seed(iteration)
        random.seed(iteration)
        location_utility = EthnicalLocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
        points_of_interest = location_utility.generate_ethnical_points_of_interest(area)
        block_locations = location_utility.read_block_locations(BLOCK_LOCATION_FILE)
        location_utility.add_ethnicity()
        location_utility.generate(points_of_interest, block_locations, set_variance=False)
        
        solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, location_utility)
        solver.calculate(has_ethnicity=False)
        integer_results.append(value(solver._prob.objective))
        x_vars = [v.varValue for v in solver._prob.variables()]
        integer_expected.append(calculate_expected_utility_wrt_ethnic(x_vars, location_utility._ethnic_agents,
                                                                      location_utility._utility))
  
        np.random.seed(iteration)
        random.seed(iteration)
        location_utility = EthnicalLocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
        points_of_interest = location_utility.generate_ethnical_points_of_interest(area)
        block_locations = location_utility.read_block_locations(BLOCK_LOCATION_FILE)
        location_utility.add_ethnicity()
        location_utility.generate(points_of_interest, block_locations, set_variance=False)
        
        solver = LotterySolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, location_utility)
        x_vars, optimal_value = solver.calculate(has_ethnicity=False)
        lottery_results.append(optimal_value)
        lottery_expected.append(calculate_expected_utility_wrt_ethnic(x_vars, location_utility._ethnic_agents,
                                                                      location_utility._utility))
        
                                                                      
    average_integer_result = np.sum(integer_results) / numb_iterations
    average_lottery_result = np.sum(lottery_results) / numb_iterations

    print ('Integer model')
    print ('Average optimal: ', average_integer_result)
    print (integer_results)
    print (integer_expected)
    print ('Lottery model')
    print ('Average optimal: ', average_lottery_result)
    print (lottery_results)
    print (lottery_expected)
    
def compare_model_constrained_location(numb_iterations=10):
    integer_results = []
    lottery_results = []
    integer_expected = []
    lottery_expected = []

    for iteration in range(numb_iterations):
        print ('Iteration:', iteration + 1)
        np.random.seed(iteration)
        random.seed(iteration)
        location_utility = EthnicalLocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
        points_of_interest = location_utility.generate_ethnical_points_of_interest(area)
        block_locations = location_utility.read_block_locations(BLOCK_LOCATION_FILE)
        location_utility.add_ethnicity()
        location_utility.generate(points_of_interest, block_locations, set_variance=False)
        
        solver = IntegerProgrammingSolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, location_utility)
        solver.add_ethnicity(location_utility, read_from_file=True, file_name=ETHNIC_FILE)
        solver.calculate(has_ethnicity=True)
        integer_results.append(value(solver._prob.objective))
        x_vars = [v.varValue for v in solver._prob.variables()]
        integer_expected.append(calculate_expected_utility_wrt_ethnic(x_vars, location_utility._ethnic_agents,
                                                                      location_utility._utility))
  
        np.random.seed(iteration)
        random.seed(iteration)
        location_utility = EthnicalLocationUtility(NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS_PER_BLOCK)
        area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
        points_of_interest = location_utility.generate_ethnical_points_of_interest(area)
        block_locations = location_utility.read_block_locations(BLOCK_LOCATION_FILE)
        location_utility.add_ethnicity()
        location_utility.generate(points_of_interest, block_locations, set_variance=False)
        
        solver = LotterySolver(NUMB_AGENTS, NUMB_BLOCKS,
                                                 NUMB_FLATS_PER_BLOCK, location_utility)
        solver.add_ethnicity(location_utility, read_from_file=True, file_name=ETHNIC_FILE)
        x_vars, optimal_value = solver.calculate(has_ethnicity=True)
        lottery_results.append(optimal_value)
        lottery_expected.append(calculate_expected_utility_wrt_ethnic(x_vars, location_utility._ethnic_agents,
                                                                      location_utility._utility))
        
                                                                      
    average_integer_result = np.sum(integer_results) / numb_iterations
    average_lottery_result = np.sum(lottery_results) / numb_iterations

    print ('Integer model')
    print ('Average optimal: ', average_integer_result)
    print (integer_results)
    print (integer_expected)
    print ('Lottery model')
    print ('Average optimal: ', average_lottery_result)
    print (lottery_results)
    print (lottery_expected)

if __name__ == '__main__':
    # turn stochastic process into deterministic, but turn off when experimenting
    np.random.seed(0) 
    random.seed(0)
    start_time = time.time()
    # set global parameter
    #set_global_parameter_test()
    set_global_parameter_actual()
    
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
    #optimal_value = model_list['unconstrained_ipsolver_random']()
    #optimal_value = model_list['constrained_ipsolver_random']()
    #optimal_value = model_list['unconstrained_ipsolver_random'](True)
    #optimal_value = model_list['constrained_ipsolver_random'](True)
    #compare_model_ip_random(10)

    #optimal_value = model_list['unconstrained_ipsolver_location']()
    #optimal_value = model_list['constrained_ipsolver_location']()
    #optimal_value = model_list['unconstrained_ipsolver_location'](True)
    #optimal_value = model_list['constrained_ipsolver_location'](True)
    #compare_model_ip_location(10)
    
    #optimal_value = model_list['unconstrained_lotterysolver_random']()
    #optimal_value = model_list['constrained_lotterysolver_random']()
    #optimal_value = model_list['unconstrained_lotterysolver_random'](True)
    #optimal_value = model_list['constrained_lotterysolver_random'](True)
    #compare_model_lottery_random(10)
    
    #optimal_value = model_list['unconstrained_lotterysolver_location']()
    #optimal_value = model_list['constrained_lotterysolver_location']()
    #optimal_value = model_list['unconstrained_lotterysolver_location'](True)
    #optimal_value = model_list['constrained_lotterysolver_location'](True)
    #compare_model_lottery_location(10)
    
    #compare_model_unconstrained_random(10)
    #compare_model_constrained_random(10)
    #compare_model_unconstrained_location(10)
    #compare_model_constrained_location(10)
    print("--- %s seconds ---" % (time.time() - start_time))