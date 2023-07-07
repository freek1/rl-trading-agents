import os

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
    n_agents, agent_type, move_prob, save_to_file, run_nr, run_time, enable_rendering = arg

    grid_width, grid_height, cell_size = get_grid_params()
    screen_width = grid_width * cell_size
    screen_height = grid_height * cell_size

    if enable_rendering:
        pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height))
        
    fps = 1
    clock = pygame.time.Clock()
    time = 1
    duration = 1000

    # Creating market squares
    market_size = 6
    market = np.full((grid_height, grid_width), False, dtype=bool)
    for x in range(int((grid_width / 2) - market_size), int((grid_width / 2) + market_size)):
            for y in range(int((grid_height / 2) - market_size), int((grid_height / 2) + market_size)):
                market[x][y] = True

    # resources
    resource_a = np.zeros((grid_width, grid_height))
    resource_b = np.zeros((grid_width, grid_height))
    max_resource_a = 2
    max_resource_b = 2

    resource_a_cell_count = 0
    resource_b_cell_count = 0
    
    # resources in sides position
    for x in range(0, grid_width):
        for y in range(0, 8):
            if not market[x][y]:
                resource_b_cell_count += 1
                resource_a[x][y] = max_resource_a  # random.uniform(m_in_w_oo_d, m_ax_w_oo_d)
    for x in range(0, grid_height):
        for y in range(32, grid_height):
            if not market[x][y]:
                resource_a_cell_count += 1
                resource_b[x][y] = max_resource_b  # random.uniform(m_in_f_oo_d, m_ax_f_oo_d)

    # resource settings
    total_resource_a_regen = 4.2
    total_resource_b_regen = 4.2
    initial_resource_a_qty = 420
    initial_resource_b_qty = 420
    initial_resource_a_qty_cell = initial_resource_a_qty / resource_a_cell_count
    initial_resource_b_qty_cell = initial_resource_b_qty / resource_b_cell_count

    gather_amount = 1.0
    # normalize resource regeneration such that the total regen. is same regardless of number of resource cells)
    resource_a_regen_rate = total_resource_a_regen / resource_a_cell_count
    resource_b_regen_rate = total_resource_b_regen / resource_b_cell_count

    # normalize initial quantities
    for x in range(0, grid_width):
        for y in range(0, grid_height):
            if resource_a[x][y] > 0:
                resource_a[x][y] = initial_resource_a_qty_cell
            if resource_b[x][y] > 0:
                resource_b[x][y] = initial_resource_b_qty_cell

    resources = {
        "resource_b": resource_b,
        "resource_a": resource_a,
    }
    max_resources = copy.deepcopy(resources)

    # setting up agents
    agents = []
    agent_positions = []

    # creating agents
    for i in range(n_agents):
        x = random.randint(0, grid_width - 2)
        y = random.randint(0, grid_height - 2)
        color = (255,110,0)
        agent = Agent(i, x, y, agent_type, color, market)
        agents.append(agent)

        # save agent position for the k_d-tree
        agent_positions.append([x, y])

    positions_tree = KDTree(agent_positions)
    alive_times = np.zeros([n_agents])
    alive_times.fill(duration)

    # run the simulation
    running = True
    
    # pack arguments
    args = enable_rendering, agents, agent_positions, resources, gather_amount, market, move_prob, alive_times, resource_a_regen_rate, resource_b_regen_rate, max_resources, screen, resource_a, resource_b, initial_resource_a_qty_cell, initial_resource_b_qty_cell, screen_width, screen_height, clock, fps, positions_tree, time

    while running:
        running, _ = run_sim_step([2, 2], args)
        


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
            
            x, y = agent.get_pos()
            agent_positions[i] = [x, y]
            states.append([x, y, agent.current_stock["resource_a"], agent.current_stock["resource_b"]])


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
            agent_positions.remove(death_agent.get_pos())
        
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
        pygame.display.flip()

    clock.tick(fps)
    time += 1

    # handle events
    for event in pygame.event.get():
        # return running = False or True
        if event.type == pygame.QUIT:
            print('test')
            return (False, states)
        else:
            print('test2')
            return (True, states)

if __name__ == "__main__":
    # n_agents, agent_type, move_prob, save_to_file, run_nr, run_time, enable_rendering
    arg = (2, 'neural_agent', 1, False, 1, 1000, True)
    run_simulation(arg)