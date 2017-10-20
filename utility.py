'''This is classes for Utility Generation'''
import matplotlib.pyplot as plt
import numpy as np
import random
import abc

'''
Abstract class Utility
'''
class Utility(abc.ABC):

    @abc.abstractmethod
    def __init__(self, numb_agents, numb_blocks, numb_flats_per_block):
        self._utility = None
    
    @abc.abstractmethod
    def generate(self):
        pass
    
    def plot(self, utility, title):
        count, bins, ignored = plt.hist(utility, bins = 'auto')
        plt.title(title)
        plt.show()

        
        
'''
Random Utility class 
'''              
class RandomUtility(Utility):

    LOWER_BOUND, UPPER_BOUND = 0.5, 2. # range of alpha & beta of BETA DISTRIBUTION
    ACTUAL_RATIO = {'CHINESE': .77, 'MALAYS': .14, 'INDIANS': .08}

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
            
            # normalize uitilities
            normalized_utilities = rough_utility_all_flat / np.sum(rough_utility_all_flat)
            utility_of_agents[agent_index] = normalized_utilities
        
        self._utility = utility_of_agents
        return utility_of_agents

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
        
    def plot_all(self):
        self.plot(self._utility, 
        title = 'Histogram of utility generated from Beta Distribution')
       
    def plot_per_block(self):
        for block_index in range(self._numb_blocks):
            start_index = block_index * self._numb_flats_per_block[block_index]
            end_index = (block_index + 1) * self._numb_flats_per_block[block_index]
            self.plot(self._utility[:, start_index:end_index],
                         'Block ' + str(block_index+1))


def test1(should_plot = False):
    np.random.seed(0) 
    random.seed(0)
    numb_agents = 50
    numb_blocks = 5
    numb_flats_per_block = [10, 10, 10, 10, 10]
    utility = RandomUtility(numb_agents, numb_blocks, numb_flats_per_block)
    utility.generate()
    
    if (should_plot == True):
        utility.plot_all()
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
        utility.plot_all()
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
    
if __name__ == '__main__':
    ''' Test suite for RandomUtility '''
    test1(False)
    test2(False)
    test3(False)
    test4(False)
    #test1(True)
    #test2(True)
    #test3(True)
    #test4(True)
    