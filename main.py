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

def simulate_and_evaluate(organism, n_agents, n_steps):
    """
    Simulate neural agent with Organism brain and return the fitness score
    """

    # n_agents, agent_type, move_prob, save_to_file, run_nr, run_time, enable_rendering
    arg = (n_agents, 'neural_agent', 1, False, 1, 500, False)
    args = setup_sim(arg)

    X = np.zeros((n_agents, 4))
    _, states = run_sim_step([2, 2], args)

    fitness = 0
    for step in range(n_steps):
        X = np.array(states)
        moves = organism.predict(X)
        move = get_move(moves)

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
    results = []
    for gen in range(generations):
        start = time.time()
        best = process_generation(gen)
        results.append(best)
        end = time.time()
        print(f'Generation {gen} finished in {end - start}s.')
    return results

def exec_parallel(generations):
    start = time.time()
    cores = multiprocessing.cpu_count()
    print(f'cores: {cores}')
    pool = multiprocessing.Pool(processes=cores*2)
    results = pool.map(process_generation, range(generations))
    end = time.time()
    print(f'Generations parallel finished in {end - start}s.')
    pool.close()
    pool.join()
    return results

# The function to create the initial population
organism_creator = lambda : Organism([4, 32, 64, 32, 4], output='softmax')
# Ecosystem requires a function that maps an organism to a real number fitness
scoring_function = lambda organism : simulate_and_evaluate(organism, n_agents=2, n_steps=20)
# Create the ecosystem
ecosystem = Ecosystem(organism_creator, scoring_function, 
                        population_size=200, holdout=0.1, mating=True)

if __name__ == '__main__':
    generations = 201

    start = time.time()
    best_organism_scores_parallel = exec_parallel(generations)
    end = time.time()
    print('Total parallel time: ', end - start)
    # Sides:
    # 2 gens: 219 total, fitness = 57
    # 10 gens: 220 total, fitness = 61
    # 16 gens, 300 pop, Organism[4, 32, 64, 32, 4]: 220 total, fitness = 58
    # 16 gens, 300 pop, Organism[4, 32, 64, 128, 64, 32, 4]: 646 total, fitness = 61
    #   -> limited by RAM 
    # 201 gens, 200 pop, [4, 32, 64, 32, 4], 4580 total, fitness = 34

    # Random Grid:
    # 16 gens, 300 pop, Organism[4, 32, 32, 4]: 655 total, fitness = 31
    # 

    n_steps = 20
    random_agent_fitness = 0
    n_agents = 2

    for step in range(n_steps):
        moves = [random.randint(0,3)] * n_agents
        move = get_move(moves)

        arg = (n_agents, 'random', 1, False, 1, 500, False)
        args = setup_sim(arg)

        _, states = run_sim_step(move, args)
        states = np.array(states)
        for i in range(n_agents):
            # Fitness is the sum of resource_a and resource_b.
            # If the agents are dead, their fitness is 0.
            if i < states.shape[0]:
                random_agent_fitness += states[i][2] + states[i][3]
        random_agent_fitness /= n_agents

    plt.plot(best_organism_scores_parallel, label=f'neural agent parallel \n {best_organism_scores_parallel[-1]}')
    plt.plot(np.arange(generations), [random_agent_fitness] * generations, label=f'random agent \n {random_agent_fitness}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness score')
    plt.title('Fitness score evolution')
    plt.legend()
    plt.show()
