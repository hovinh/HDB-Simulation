import numpy as np
import random 
import matplotlib.pyplot as plt

NUMB_AGENTS = 50
NUMB_BLOCKS = 5
NUMB_FLATS_PER_BLOCK = 10
NUMB_TOTAL_FLATS = 50
LOWER_BOUND, UPPER_BOUND = 0.5, 2. # range of alpha & beta of BETA DISTRIBUTION

def generate_beta_distribution_per_block():
    beta_distribution_per_block = [(0., 0.) for i in range(NUMB_BLOCKS)]
    
    for block_index in range(NUMB_BLOCKS):
        
        # generate parameter alpha & beta
        alpha = random.uniform(LOWER_BOUND, UPPER_BOUND)
        beta = random.uniform(LOWER_BOUND, UPPER_BOUND)
        
        beta_distribution_per_block[block_index] = (alpha, beta)
    
    return beta_distribution_per_block
    
def generate_agent_utility_beta_distribution(beta_distribution_per_block):
    utility_of_agents = np.empty([NUMB_AGENTS, NUMB_TOTAL_FLATS])
    
    for agent_index in range(NUMB_AGENTS):
        
        rough_utility_all_flat = np.array([])
        
        for block_index in range(NUMB_BLOCKS):
        
            alpha, beta = beta_distribution_per_block[block_index] 
        
            # generate uitility drawn from BETA DISTRIBUTION
            rough_utilities = np.random.beta(alpha, beta, NUMB_FLATS_PER_BLOCK)
            rough_utility_all_flat = np.concatenate((rough_utility_all_flat,
                                                         rough_utilities))
        
        # normalize uitilities
        normalized_utilities = rough_utility_all_flat / np.sum(rough_utility_all_flat)
        
        utility_of_agents[agent_index] = normalized_utilities
    
    return utility_of_agents

def plot_utility(utilities, title):
    count, bins, ignored = plt.hist(utilities, bins = 'auto')
    plt.title(title)
    plt.show()
    
def generate_utility_randomly():
    '''
    Given NUMB_AGENTS, NUMB_BLOCKS, NUMB_FLATS, one would generate utility of 
    each agents distributed for all flats from a prefefined distribution.
    
    Algorithm:
    - For each block, generate a lambda from NORMAL DISTRIBUTION for each flat
    in decreasing order
    - For each agent, draw utility from POISSON DISTRIBUTION with generated lambda
    - Normalize all uitility such that non-negative value and total value is 1 
    
    '''
    
    beta_distribution_per_block = generate_beta_distribution_per_block()
    utility_of_agents = generate_agent_utility_beta_distribution(
                                beta_distribution_per_block)
    plot_utility(utility_of_agents, 'Histogram of utility generated from Beta Distribution')
    for block_index in range(NUMB_BLOCKS):
        start_index = block_index * NUMB_FLATS_PER_BLOCK
        end_index = (block_index + 1) * NUMB_FLATS_PER_BLOCK
        plot_utility(utility_of_agents[:, start_index:end_index], 'Block ' + str(block_index))

if __name__ == '__main__':
    # turn stochastic process into deterministic, but turn off when experimenting
    np.random.seed(0) 
    random.seed(0)
    
    a = generate_utility_randomly()
    
    
    
''' 
def test():
    mu, sigma = 0.5, 0.1 # mean and standard deviation
    s = np.random.normal(mu, sigma, 1000)
    count, bins, ignored = plt.hist(s, 30, normed=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
             np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
            linewidth=2, color='r')
    plt.show()
    
'''