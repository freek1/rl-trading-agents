import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from Organism import Organism
from Ecosystem import Ecosystem

from simulation import *
from funcs import *

from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

# The function to create the initial population
organism_creator = lambda : Organism([4, 16, 16, 16, 4], output='softmax')


def simulate_and_evaluate(organism, n_agents):
    """
    Randomly generate `replicates` samples in [0,1],
    use the organism to predict their corresponding value,
    and return the fitness score of the organism
    """

    # n_agents, agent_type, move_prob, save_to_file, run_nr, run_time, enable_rendering
    arg = (n_agents, 'neural_agent', 1, False, 1, 1000, False)
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
        fitness += states[i][2] + states[i][3]

    return fitness

# Ecosystem requires a function that maps an organism to a real number fitness
scoring_function = lambda organism : simulate_and_evaluate(organism, n_agents=50)
# Create the ecosystem
ecosystem = Ecosystem(organism_creator, scoring_function, 
                      population_size=100, holdout=0.1, mating=True)
# Save the fitness score of the best organism in each generation
best_organism_scores = [ecosystem.get_best_organism(include_reward=True)[1]]


def process_generation(i):
    print('generation:', i)
    ecosystem.generation()
    this_generation_best = ecosystem.get_best_organism(include_reward=True)
    return this_generation_best[1]


if __name__ == '__main__':
    generations = 101
    pool = multiprocessing.Pool(processes=16)
    results = pool.map(process_generation, range(generations))
    pool.close()
    pool.join()

    best_organism_scores = results
    print(best_organism_scores)

    plt.plot(best_organism_scores)
    plt.xlabel('Generation')
    plt.ylabel('Fitness score')
    plt.title('Fitness score evolution')
    plt.show()
