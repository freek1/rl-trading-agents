import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from Organism import Organism
from Ecosystem import Ecosystem

from simulation import *
from funcs import *

from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
import time

# The function to create the initial population
organism_creator = lambda : Organism([4, 16, 32, 16, 4], output='softmax')
# Ecosystem requires a function that maps an organism to a real number fitness
scoring_function = lambda organism : simulate_and_evaluate(organism, n_agents=2)
# Create the ecosystem
ecosystem = Ecosystem(organism_creator, scoring_function, 
                        population_size=200, holdout=0.1, mating=True)

save_all_moves = []

def simulate_and_evaluate(organism, n_agents):
    """
    Simulate neural agent with Organism brain and return the fitness score
    """

    # n_agents, agent_type, move_prob, save_to_file, run_nr, run_time, enable_rendering
    arg = (n_agents, 'neural_agent', 1, False, 1, 500, False)
    args = setup_sim(arg)

    X = np.zeros((n_agents, 4))
    _, states = run_sim_step([2, 2], args)

    n_steps = 20
    
    fitness = 0
    for step in range(n_steps):
        X = np.array(states)
        moves = organism.predict(X)
        move = get_move(moves)

        save_all_moves.append(move)

        _, states = run_sim_step(move, args)
        states = np.array(states)
        for i in range(n_agents):
            # Fitness is the sum of resource_a and resource_b.
            # If the agents are dead, their fitness is 0.
            if i < states.shape[0]:
                fitness += states[i][2] + states[i][3]
        fitness /= n_agents

    return fitness

def process_generation(i):
    print('Running generation', i)
    ecosystem.generation()
    this_generation_best = ecosystem.get_best_organism(include_reward=True)
    return this_generation_best[1]

def exec_series(generations):
    start = time.time()
    results = []
    for gen in range(generations):
        best = process_generation(gen)
        results.append(best)
    end = time.time()
    print(f'Generation {i} finished in {end-start}s.')
    return results

def exec_parallel(generations):
    start = time.time()
    cores = multiprocessing.cpu_count()
    print(f'cores: {cores}, spawning {np.min(int(cores*1.25), generations)} workers')
    pool = multiprocessing.Pool(processes = np.min(int(cores*1.25), generations))
    results = pool.map(process_generation, range(generations))
    end = time.time()
    print(f'Generations parallel finished in {end-start}s.')
    pool.close()
    pool.join()
    return results

if __name__ == '__main__':
    generations = 2

    start = time.time()
    best_organism_scores_series = exec_series(generations)
    end = time.time()
    print('Total series time: ', end - start)

    start = time.time()
    best_organism_scores_parallel = exec_parallel(generations)
    end = time.time()
    print('Total parallel time: ', end - start)

    n_steps = 10
    random_agent_fitness = 0

    for step in range(n_steps):
        moves = [random.randint(0,3), random.randint(0,3)]
        move = get_move(moves)

        arg = (2, 'random', 1, False, 1, 500, False)
        args = setup_sim(arg)

        _, states = run_sim_step(move, args)
        states = np.array(states)
        for i in range(2):
            # Fitness is the sum of resource_a and resource_b.
            # If the agents are dead, their fitness is 0.
            if i < states.shape[0]:
                random_agent_fitness += states[i][2] + states[i][3]
        random_agent_fitness /= 2

    plt.plot(best_organism_scores_series, label='neural agent series')
    plt.plot(best_organism_scores_parallel, label='neural agent parallel')
    plt.plot(np.arange(generations), [random_agent_fitness] * generations, label='random agent')
    plt.xlabel('Generation')
    plt.ylabel('Fitness score')
    plt.title('Fitness score evolution')
    plt.show()
