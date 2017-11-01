'''This is classes for Utility Generation'''
import matplotlib.pyplot as plt
import numpy as np
import random
import abc
import math

'''
Abstract class Utility
'''
class Utility(abc.ABC):
    ACTUAL_RATIO = {'CHINESE': .77, 'MALAYS': .14, 'INDIANS': .08}
    NUMB_ETHNICS = 3
    
    @abc.abstractmethod
    def __init__(self):
        self._utility = None
    
    @abc.abstractmethod
    def generate(self):
        pass
            
    def add_ethnicity(self):
        # Assign randomly ethnic to agents according to actual proportion over population
        numb_chinese = int(self._numb_agents * self.ACTUAL_RATIO['CHINESE'])
        numb_malays = int(self._numb_agents * self.ACTUAL_RATIO['MALAYS'])
        numb_indians = self._numb_agents - numb_chinese - numb_malays 
            
        ethnic_agents = np.full((1, numb_chinese), 'C')[0]
        ethnic_agents = np.concatenate((ethnic_agents, np.full((1, numb_malays), 'M')[0]),
                                       axis = 0)
        ethnic_agents = np.concatenate((ethnic_agents, np.full((1, numb_indians), 'I')[0]),
                                       axis = 0)
        ethnic_agents = np.random.permutation(ethnic_agents)
        self._ethnic_agents = ethnic_agents 
        return ethnic_agents

    def generate_uniform(self, numb_agents, numb_total_flats):
        '''
        Generate utility equally among flats
        @param:
            numb_agents: integer
            numb_total_flats: integer
        @return:
            utility_of_agents: np.ndarray 2 dimensions with shape = (numb_agents, numb_total_flats)
        '''
        
        utility_of_agents = np.full((numb_agents, numb_total_flats), 1. / numb_total_flats)
        self._utility = utility_of_agents
        return utility_of_agents
        
    def plot(self, utility, title):
        count, bins, ignored = plt.hist(np.ndarray.flatten(utility), bins = 'auto')
        plt.title(title)
        plt.show()

    def plot_all(self, title):
        self.plot(self._utility, title = title)
       
    def plot_per_block(self):
        cur_block_index = 0
        for block_index in range(self._numb_blocks):
            start_index = cur_block_index
            end_index = cur_block_index + self._numb_flats_per_block[block_index] 
            self.plot(self._utility[:, start_index:end_index],
                         'Block ' + str(block_index+1))
            cur_block_index = end_index
            
        
'''
Random Utility class 
'''              
class RandomUtility(Utility):

    LOWER_BOUND, UPPER_BOUND = 0.5, 2. # range of alpha & beta of BETA DISTRIBUTION

    def __init__(self, numb_agents, numb_blocks, numb_flats_per_block):
        self._numb_agents = numb_agents
        self._numb_blocks = numb_blocks
        self._numb_flats_per_block = numb_flats_per_block
        self._numb_total_flats = np.sum(numb_flats_per_block)
        self._utility = None
        
    def generate_beta_distribution_per_block(self):
        beta_distribution_per_block = [(0., 0.) for i in range(self._numb_blocks)]
        
        for block_index in range(self._numb_blocks):
            
            # generate parameter alpha & beta
            alpha = random.uniform(self.LOWER_BOUND, self.UPPER_BOUND)
            beta = random.uniform(self.LOWER_BOUND, self.UPPER_BOUND)
            
            beta_distribution_per_block[block_index] = (alpha, beta)
        
        self._beta_distribution_per_block = beta_distribution_per_block
        return beta_distribution_per_block
        
    def generate(self):
        '''
        Generate utility according to BETA DISTRIBUTION
        @return:
            utility_of_agents: np.ndarray 2 dimensions with shape = (numb_agents, numb_total_flats)
        '''
        
        self.generate_beta_distribution_per_block()

        utility_of_agents = np.empty([self._numb_agents, self._numb_total_flats])
        
        for agent_index in range(self._numb_agents):
            
            rough_utility_all_flat = np.array([])
            
            for block_index in range(self._numb_blocks):
            
                alpha, beta = self._beta_distribution_per_block[block_index] 
            
                # generate uitility drawn from BETA DISTRIBUTION
                rough_utilities = np.random.beta(alpha, beta, self._numb_flats_per_block[block_index])
                rough_utility_all_flat = np.concatenate((rough_utility_all_flat,
                                                             rough_utilities))
            
            # normalize utilities
            normalized_utilities = rough_utility_all_flat / np.sum(rough_utility_all_flat)
            utility_of_agents[agent_index] = normalized_utilities
        
        self._utility = utility_of_agents
        return utility_of_agents

def test1(should_plot = False):
    np.random.seed(0) 
    random.seed(0)
    numb_agents = 50
    numb_blocks = 5
    numb_flats_per_block = [10, 10, 10, 10, 10]
    utility = RandomUtility(numb_agents, numb_blocks, numb_flats_per_block)
    utility.generate()
    
    if (should_plot == True):
        utility.plot_all('Histogram of utility generated from Beta Distribution')
        utility.plot_per_block()

def test1_1(should_plot = False):
    np.random.seed(0) 
    random.seed(0)
    numb_agents = 50
    numb_blocks = 5
    numb_flats_per_block = [10, 10, 10, 10, 10]
    utility = RandomUtility(numb_agents, numb_blocks, numb_flats_per_block)
    utility.generate_uniform(numb_agents, numb_agents)
    
    if (should_plot == True):
        utility.plot_all('Histogram of utility generated from Beta Distribution')
        utility.plot_per_block()
    
def test2(should_plot = False):
    np.random.seed(0) 
    random.seed(0)
    numb_agents = 5000
    numb_blocks = 5
    numb_flats_per_block = [10, 10, 10, 10, 10]
    utility = RandomUtility(numb_agents, numb_blocks, numb_flats_per_block)
    utility.generate()
    
    if (should_plot == True):
        utility.plot_all('Histogram of utility generated from Beta Distribution')
        utility.plot_per_block()

def test3(should_plot = False):
    np.random.seed(0) 
    random.seed(0)
    numb_agents = 50
    numb_blocks = 5
    numb_flats_per_block = [10, 10, 10, 10, 10]
    utility = RandomUtility(numb_agents, numb_blocks, numb_flats_per_block)
    print (utility._utility)
        
def test4(should_plot = False):
    np.random.seed(0) 
    random.seed(0)
    numb_agents = 100
    numb_blocks = 5
    numb_flats_per_block = [10, 10, 10, 10, 10]
    utility = RandomUtility(numb_agents, numb_blocks, numb_flats_per_block)
    utility.add_ethnicity()
    utility.generate()
    print (utility._ethnic_agents)
 
'''
Location-based Utility class 
'''              
class LocationUtility(Utility):

    def __init__(self, numb_agents, numb_blocks, numb_flats_per_block):
        self._numb_agents = numb_agents
        self._numb_blocks = numb_blocks
        self._numb_flats_per_block = numb_flats_per_block
        self._numb_total_flats = np.sum(numb_flats_per_block)
        self._utility = None

    def generate_points_of_interest(self, area):
        '''
        Generate 1 point of interest for every agent
        @param:
            area: limitation of where points is generate
                 tuple of 4: (lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y)
        @return:
            points_of_interest: list of 2-element tuples(coordinate) of each agent, shape = (numb_agents, 2)
        '''
        
        lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = area
        random_x = np.random.uniform(lower_bound_x, upper_bound_x, self._numb_agents)
        random_y = np.random.uniform(lower_bound_y, upper_bound_y, self._numb_agents)
        points_of_interest = [(random_x[i], random_y[i]) for i in range(self._numb_agents)]
        self._points_of_interest = points_of_interest
        
        return points_of_interest
    
    def generate(self, points_of_interest, block_locations):
        '''
        Generate utility according to LOCATION
        @param:
            points_of_interest: list of 2-element tuples(coordinate) of each agent, shape = (numb_agents, 2)
            block_locations: list of 2-element tuples(coordinate) of each block
        @return:
            utility_of_agents: np.ndarray 2 dimensions with shape = (numb_agents, numb_total_flats)
        '''
        
        utility_of_agents = np.empty([self._numb_agents, self._numb_total_flats])
        
        for agent_index in range(self._numb_agents):
            
            rough_utility_all_flat = np.array([])
            agent_x, agent_y = points_of_interest[agent_index]
            
            for block_index in range(self._numb_blocks):
            
                # utility is inversely proportional to distance to block
                # every flat in the same block having same utility
                block_x, block_y = block_locations[block_index]
                distance_to_block = self.calculate_distance_to_block(agent_x, agent_y, block_x, block_y)
                block_utility = 1. / distance_to_block
                rough_utilities = np.array([block_utility] * self._numb_flats_per_block[block_index])
                rough_utility_all_flat = np.concatenate((rough_utility_all_flat,
                                                             rough_utilities))
            
            # normalize utilities
            normalized_utilities = rough_utility_all_flat / np.sum(rough_utility_all_flat)
            utility_of_agents[agent_index] = normalized_utilities
        
        self._utility = utility_of_agents
        return utility_of_agents
    
    def calculate_distance_to_block(self, x, y, block_x, block_y):
        distance = math.sqrt((x - block_x)**2 + (y - block_y)**2)
        return distance
        
    def read_block_locations(self, file_name = 'test_location.txt'):
        block_location_list = []
        with open(file_name, 'r') as file:
            for line in file:
                line = line.strip()
                block_location_list.append([float(i) for i in line.split(',')])
          
        self._block_location_list = block_location_list
        return block_location_list

def test5(should_plot = False):
    np.random.seed(0) 
    random.seed(0)
    numb_agents = 50
    numb_blocks = 5
    numb_flats_per_block = [10, 10, 10, 10, 10]
    utility = LocationUtility(numb_agents, numb_blocks, numb_flats_per_block)
    
    area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
    points_of_interest = utility.generate_points_of_interest(area)
    '''
    block_locations = [(103.8, 1.31), (103.9, 1.31), (103.7, 1.35),
                       (103.85, 1.4), (103.9, 1.37)]
    '''
    block_locations = utility.read_block_locations()
    print (block_locations)
    utility.generate(points_of_interest, block_locations)
    
    if (should_plot == True):
        utility.plot_all('Histogram of utility generated from Block Location')
        utility.plot_per_block()            

def test6(should_plot = False):
    np.random.seed(0) 
    random.seed(0)
    numb_agents = 50
    numb_blocks = 5
    numb_flats_per_block = [10, 10, 10, 10, 10]
    utility = LocationUtility(numb_agents, numb_blocks, numb_flats_per_block)
    
    area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
    points_of_interest = utility.generate_points_of_interest(area)
    block_locations = [(103.8, 1.31), (103.9, 1.31), (103.7, 1.35),
                       (103.85, 1.4), (103.9, 1.37)]
    utility.add_ethnicity()
    utility.generate(points_of_interest, block_locations)
    
    if (should_plot == True):
        utility.plot_all('Histogram of utility generated from Block Location')
        utility.plot_per_block()     
 
'''
Location-based Ethinic Group Utility class 
'''              
class EthnicalLocationUtility(Utility):

    def __init__(self, numb_agents, numb_blocks, numb_flats_per_block):
        self._numb_agents = numb_agents
        self._numb_blocks = numb_blocks
        self._numb_flats_per_block = numb_flats_per_block
        self._numb_total_flats = np.sum(numb_flats_per_block)
        self._utility = None

    def convert_to_beta_distribution_parameters(self, mean, variance):
        '''
        Source: https://stats.stackexchange.com/questions/142626/mean-and-variance-of-a-beta-distribution-with-alpha-ge-1-or-beta-ge-1
        @param:
            mean: 0 < mean < 1
            variance: 0 < variance < mean * (1 - mean)
        @ return
            alpha, beta of BETA DISTRIBUTION
        '''
        alpha = mean * ((mean * (1 - mean) - variance) / variance)
        beta = (1 - mean) * ((mean * (1 - mean) - variance) / variance)
        return alpha, beta
        
    def ethnical_beta_distribution(self, ethnic_distances, numb_blocks, set_variance = None):
        beta_distributions = {}
        ethnic_list = ['C', 'M', 'I']
        
        # we need max_mean to normalize into range [0,1]
        min_distance = ethnic_distances['C'][0]
        for ethnicity in ethnic_list:
            distance = min(ethnic_distances[ethnicity])
            if (distance < min_distance):
                min_distance = distance

        max_mean = (1. / min_distance) + 1.        
        
        for ethnic_index in range(self.NUMB_ETHNICS):
            
            ethnic = ethnic_list[ethnic_index]
            parameters_list = []
            for block_index in range(numb_blocks):
                mean = 1. / ethnic_distances[ethnic][block_index]
                mean = mean / max_mean
                variance = 0.005
                if (set_variance == False):
                    variance = mean * (1 - mean) * 0.5
                alpha, beta = self.convert_to_beta_distribution_parameters(mean, variance)
                parameters_list.append([alpha, beta])
                
            beta_distributions[ethnic] = parameters_list

        self._beta_distributions = beta_distributions
        return beta_distributions
        
    def generate_ethnical_points_of_interest(self, area):
        '''
        Generate 1 point of interest for every group of ethnicicty
        @param:
            area: limitation of where points is generate
                 tuple of 4: (lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y)
        @return:
            points_of_interest: list of 2-element tuples(coordinate) of ethnicity, shape = (numb_agents, 2)
        '''
        
        lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y = area
        random_x = np.random.uniform(lower_bound_x, upper_bound_x, self._numb_agents)
        random_y = np.random.uniform(lower_bound_y, upper_bound_y, self._numb_agents)
        points_of_interest = [(random_x[i], random_y[i]) for i in range(self.NUMB_ETHNICS)]
        self._points_of_interest = points_of_interest
        
        return points_of_interest
    
    def generate(self, points_of_interest, block_locations, set_variance = False):
        '''
        Generate utility according to LOCATION of ETHNICITY
        @param:
            points_of_interest: list of 2-element tuples(coordinate) of each agent, shape = (numb_agents, 2)
            block_locations: list of 2-element tuples(coordinate) of each block
        @return:
            utility_of_agents: np.ndarray 2 dimensions with shape = (numb_agents, numb_total_flats)
        '''
        
        utility_of_agents = np.empty([self._numb_agents, self._numb_total_flats])
        ethnic_distances = self.calculate_distance_ethnic_to_block(block_locations)
        ethnic_beta_distributions = self.ethnical_beta_distribution(ethnic_distances, self._numb_blocks, set_variance)
        
        for agent_index in range(self._numb_agents):
            
            rough_utility_all_flat = np.array([])
            ethnic_agent = self._ethnic_agents[agent_index]            
            for block_index in range(self._numb_blocks):
            
                alpha, beta = ethnic_beta_distributions[ethnic_agent][block_index]
                rough_utilities = np.random.beta(alpha, beta, self._numb_flats_per_block[block_index])
                rough_utility_all_flat = np.concatenate((rough_utility_all_flat,
                                                             rough_utilities))
            
            # normalize utilities
            normalized_utilities = rough_utility_all_flat / np.sum(rough_utility_all_flat)
            utility_of_agents[agent_index] = normalized_utilities
        
        self._utility = utility_of_agents
        return utility_of_agents
    
    def calculate_distance_to_block(self, x, y, block_x, block_y):
        distance = math.sqrt((x - block_x)**2 + (y - block_y)**2)
        return distance
    
    def calculate_distance_ethnic_to_block(self, block_locations):
        numb_blocks = len(block_locations)
        ethnic_distances = {}
        ethnic_list = ['C', 'M', 'I']
        
        for ethnic_index in range(self.NUMB_ETHNICS):
            
            block_distances = []
            for block_index in range(numb_blocks):
                x, y = self._points_of_interest[ethnic_index]
                block_x, block_y = block_locations[block_index]
                distance = self.calculate_distance_to_block(x, y, block_x, block_y)
                block_distances.append(distance)
            
            ethnic = ethnic_list[ethnic_index]
            ethnic_distances[ethnic] = block_distances

        self._ethnic_distances = ethnic_distances
        return ethnic_distances
        
    def read_block_locations(self, file_name = 'test_location.txt'):
        block_location_list = []
        with open(file_name, 'r') as file:
            for line in file:
                line = line.strip()
                block_location_list.append([float(i) for i in line.split(',')])
          
        self._block_location_list = block_location_list
        return block_location_list

def test7(should_plot = False):
    np.random.seed(0) 
    random.seed(0)
    numb_agents = 50
    numb_blocks = 5
    numb_flats_per_block = [10, 10, 10, 10, 10]
    utility = EthnicalLocationUtility(numb_agents, numb_blocks, numb_flats_per_block)
    
    area = [103.675326, 103.913309, 1.302669, 1.424858] # actual limitation of Singapore
    points_of_interest = utility.generate_ethnical_points_of_interest(area)
    block_locations = utility.read_block_locations()
    utility.add_ethnicity(); print (utility._ethnic_agents)
    utility.generate(points_of_interest, block_locations, set_variance=False)
    
    if (should_plot == True):
        utility.plot_all('Histogram of utility generated from Ethnical Points of Interest')
        utility.plot_per_block()  
        
if __name__ == '__main__':
    ''' Test suite for RandomUtility '''
    test1(False)
    test1_1(False)
    test2(False)
    test3(False)
    test4(False)
    #test1(True)
    #test1_1(True)
    #test2(True)
    #test3(True)
    #test4(True)
    
    
    ''' Test suite for LocationUtility '''
    test5(False)
    test6(False)
    test7(False)
    #test5(True)
    #test6(True)
    #test7(True)
    