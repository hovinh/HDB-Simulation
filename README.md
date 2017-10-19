# HDB-Simulation

This project aims to simulate the Housing Development Board's allocations of Singapore. Though it is worth to mention that many research papers have pointed out the briliance of this model in comparison with others around the world, from social welfare, real estate and housings' perspective, no official documentation explains explicitly the underlying mechanism and its correctness. With this in mind, we target to answer 2 primary questions:
* Is there any gap between the current solution and the actual optimal allocation, with and without ethnicity constraint?
* Can we propose a better approach?

## Unconstrained model

One step at a time, our first attemp is simplifying the whole model into its simplest form, which consists of $N$ agents (buyers) and $M$ flats (apartments). Given utility $u_{i, j}$ of agent $i$ for each flat, playing the role of God, we will assign each pair in such a way that total utility of final allocation is maximum. This optimized problem could be simply addressed by formulating into Linear Programming form, once acknowledges the preference of agents.

Unfortunately, it surely takes months and costly effort to collect actual preference of buyer, let alone noises and errors due to human's limitation. However, thanks to openly public dataset of Singapore govenrment, we can connect reasonably a subset of information to create an acceptable utility model. By saying a subset, in fact each information can stand alone to make an appropriate estimation of actual preferences. Nevertheless, we start with 3 simple pieces of information:

1. Trend-based: People tend to buy houses in favour of current trend. In other words, a high estimatedly value house is prefered by a great proportion of population who are considering it in their buying list.

2. Flat type: Each type of family has a certain flat-type in mind. A 4-people family definitely have no interest in single-room flat, for instance.  

3. Location-based: Needless to say, living near parents, your work office, or good public school, are the most concerned factors when speaking of choosing houses. Given a point of interest of each agents, the desire for the flat inversely proportional to its distance to that particular point. We can proceed further to floor level as well as how many views one can have, but this could be left for now. 


### Trend-based utility

Simplicity provides a good start and direction for where the project is heading to. Applying the same philosophy, this utility is no other than verification of the implementation, an Input/Output testing from the point of view of Software Engineering. Another point is notably worth to say, though now showing obviously, is this model is the easiest to reduce the Computational Complexity. 

~~~
1. Assign the same BETA distribution to all flats in each block
2. For each agent_i:
    Draw the utility of her for each flats from predefined distribution
    Normalize utility such that they are non-negative and sum is equal to 1    
~~~

With algorithm as above, newly generated utility would be put into this problem formulation to solve:

$$max \sum_{j \in M} \sum_{i \in N} u_{ij}x_{ij}$$

such that 

$$\sum_{i \in N} x_{ij} \leq 1, \forall i$$
$$\sum_{j \in M} x_{ij} \leq 1, \forall j$$
$$x_{ij} \in \{0; 1\}$$

### Location-based
