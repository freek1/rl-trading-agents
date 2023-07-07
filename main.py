from Organism import Organism
from Ecosystem import Ecosystem

from simulation import *
from funcs import *


import numpy as np

# The function to create the initial population
organism_creator = lambda : Organism([4, 16, 16, 16, 4], output='softmax')


def simulate_and_evaluate(organism, n_agents=2):
    """
    Randomly generate `replicates` samples in [0,1],
    use the organism to predict their corresponding value,
    and return the fitness score of the organism
    """

    # X = [x, y, resources_a, resources_b] * agents
    arg = (n_agents, 'neural_agent', 1, False, 1, 1000, True)
    args = setup_sim(arg)

    X = np.zeros((n_agents, 4))
    _, states = run_sim_step([2, 2], args)

    for i in range(n_agents):
        X[i, :] = states[i]
    moves = organism.predict(X)
    move = get_move(moves)

    _, states = run_sim_step(move, args)

    fitness = 0
    for i in range(n_agents):
        # Fitness is the sum of all resA and resB
        fitness += states[i][3] + states[i][4]

    return fitness

# Ecosystem requires a function that maps an organism to a real number fitness
scoring_function = lambda organism : simulate_and_evaluate(organism, n_agents=2)
# Create the ecosystem
ecosystem = Ecosystem(organism_creator, scoring_function, 
                      population_size=100, holdout=0.1, mating=True)
# Save the fitness score of the best organism in each generation
best_organism_scores = [ecosystem.get_best_organism(include_reward=True)[1]]
generations = 201
for i in range(generations):
    print('generation:', i)
    ecosystem.generation()
    this_generation_best = ecosystem.get_best_organism(include_reward=True)
    best_organism_scores.append(this_generation_best[1])



