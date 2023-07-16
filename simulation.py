import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
import random
import numpy as np
import seaborn as sns
import copy
import pandas as pd
import os
from datetime import datetime
import multiprocessing
import traceback

# functions file
from funcs import *

# agent class
from agent import Agent

def run_simulation(arg):
    args = setup_sim(arg)

    running = True

    while running:
        running, states = run_sim_step([2, 2], args)

    return False, states
        

def run_sim_step(preferred_direction, args):
    # unpack arguments
    enable_rendering, agents, agent_positions, resources, gather_amount, market, move_prob, alive_times, resource_a_regen_rate, resource_b_regen_rate, max_resources, screen, resource_a, resource_b, initial_resource_a_qty_cell, initial_resource_b_qty_cell, screen_width, screen_height, clock, fps, positions_tree, time = args

    # Store states for prediction organisms
    states = [] # [x, y, resA, resB]
    # counting the nr of alive agents for automatic stopping
    nr_agents = 0

    # update the agents
    for i, agent in enumerate(agents):
        if agent.is_alive():   
            agent.update_behaviour(positions_tree, agent_positions, agents, 6, 20)
            nr_agents += 1
            agent.update_time_alive()

            if len(agent.nearest_neighbors) == 0 and agent.treshold_new_neighbours > 0:
                agent.update_treshold_new_neighbours()
                
            # update the resource gathering
            chosen_resource = choose_resource(agent, resources, gather_amount)
            take_resource(agent, chosen_resource, resources, gather_amount)

            agent.set_closest_market_pos(find_closest_market_pos(agent, market))
            agent.update_behaviour(positions_tree, agent_positions, agents, 6, 20) # to prevent two trades in same timestep

            # probabalistic movement
            if random.uniform(0, 1) < move_prob:
                if preferred_direction != [2, 2]:
                    # we get pref. direction from the Organism (AgentBrain)
                    move_agent(preferred_direction[i], agent, agents)
                else:
                    pref_direction = agent.choose_step()
                    move_agent(pref_direction, agent, agents)

            # update market bool
            agent.set_in_market(in_market(agent, market))

            # closest distance to market
            agent.set_closest_market_pos(find_closest_market_pos(agent, market))

            # update agent food locations
            set_food_locations(agent, 10, resource_a, resource_b)
            
            x, y = agent.get_pos()
            agent_positions[i] = [x, y]
            info_agent = [x, y, agent.current_stock["resource_a"], agent.current_stock["resource_b"], list(agent.get_food_locations())]
            info_agent_flat = flatten_list(info_agent)
            states.append(list(np.ravel(info_agent_flat)))

        death_agents = []
        # upkeep of agents and check if agent can survive
        for agent in agents:
            agent.upkeep()
            
            # if agent died, then remove from list and save death time
            if not agent.is_alive():
                death_agents.append(agent)
                alive_times[agent.id] = time

        for death_agent in death_agents:
            agents.remove(death_agent)
            # print(agent_positions, list(death_agent.get_pos()))
            try:
                agent_positions.remove(list(death_agent.get_pos()))
            except:
                continue
        
        if len(agent_positions) > 0:
            # updating KD-tree
            positions_tree = KDTree(agent_positions)  


        for resource in resources:  
            regen_rate = (resource_a_regen_rate if resource == "resource_a" else resource_b_regen_rate)  # get regen_rate for specific resource
            for y in range(grid_height):
                for x in range(grid_width):
                    if (resources[resource][x][y] < max_resources[resource][x][y] - regen_rate):
                        resources[resource][x][y] += regen_rate
                    else:
                        resources[resource][x][y] = max_resources[resource][x][y]  # set to max        

    if enable_rendering:
        update_screen_resources(screen, resource_a, resource_b, market, initial_resource_a_qty_cell, initial_resource_b_qty_cell)
        update_screen_agents(screen, agents)
        update_screen_grid(screen, screen_width, screen_height)

    # update the display
    if enable_rendering:
        pygame.display.flip()

    clock.tick(fps)
    time += 1

    if len(agents) == 0 or time == 500:
        print('Stopping sim: no agents or time up.')
        print(time, len(agents))
        return (False, states)

    # handle events
    if enable_rendering:
        for event in pygame.event.get():
            # return running = False or True
            if event.type == pygame.QUIT:
                return (False, states)
        return (True, states)
    else:
        return (True, states)

def task(iteration):
    arg = (2, 'neural_agent', 1, False, 1, 500, True)
    return run_simulation(arg)    

if __name__ == "__main__":
    # n_agents, agent_type, move_prob, save_to_file, run_nr, run_time, enable_rendering
    # run_simulation(arg)
    pool = multiprocessing.Pool()
    for res in pool.map(task, range(1)):
        print(res)
    pool.close()
    pool.join()