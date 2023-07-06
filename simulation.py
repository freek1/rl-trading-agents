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

    yellow = (255, 255, 0)
    dark_green = (0, 200, 0)
    blue = (30, 70, 250)
    black = (0, 0, 0)
    white = (255, 255, 255)

    grid_width, grid_height, cell_size = get_grid_params()
    screen_width = grid_width * cell_size
    screen_height = grid_height * cell_size

    if enable_rendering:
        pygame.init()
        screen = pygame.display.set_mode((screen_width, screen_height))
        
    fps = 20
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

    while running:
        # handle events
        if enable_rendering:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        # counting the nr of alive agents for automatic stopping
        nr_agents = 0

    # update the agents
        for i, agent in enumerate(agents):
            if agent.is_alive():   
                agent.update_behaviour(positions_tree, agent_positions, agents, 6, 20)
                nr_agents += 1
                agent.update_time_alive()
                x, y = agent.get_pos()


        for resource in resources:  
            regen_rate = (
                resource_a_regen_rate if resource == "resource_a" else resource_b_regen_rate
            )  # get regen_rate for specific resource
            for y in range(grid_height):
                for x in range(grid_width):
                    if (
                        resources[resource][x][y]
                        < max_resources[resource][x][y] - regen_rate
                    ):
                        resources[resource][x][y] += regen_rate
                    else:
                        resources[resource][x][y] = max_resources[resource][x][y]  # set to max        

        if enable_rendering:
            # clear the screen
            screen.fill(white)
            # draw resources
            for row in range(grid_height):
                for col in range(grid_width):
                    resource_a_value = resource_a[row][col]
                    resource_b_value = resource_b[row][col]
                    # map the resource value to a shade of brown or green
                    if market[row][col]:
                        blended_color = yellow
                    else:
                        # resource_b: g_re_en
                        inv_resource_b_color = tuple(map(lambda i, j: i - j, white, dark_green))
                        resource_b_percentage = resource_b_value / initial_resource_b_qty_cell
                        inv_resource_b_color = tuple(
                            map(lambda i: i * resource_b_percentage, inv_resource_b_color)
                        )
                        resource_b_color = tuple(map(lambda i, j: i - j, white, inv_resource_b_color))
                        resource_a: blue
                        inv_resource_a_color = tuple(map(lambda i, j: i - j, white, blue))
                        resource_a_percentage = resource_a_value / initial_resource_a_qty_cell
                        inv_resource_a_color = tuple(
                            map(lambda i: i * resource_a_percentage, inv_resource_a_color)
                        )
                        resource_a_color = tuple(map(lambda i, j: i - j, white, inv_resource_a_color))

                        # weighted blended color
                        if resource_b_percentage > 0.0 and resource_b_percentage > 0.0:
                            resource_b_ratio = resource_b_percentage / (resource_b_percentage + resource_a_percentage)
                            resource_a_ratio = resource_a_percentage / (resource_b_percentage + resource_a_percentage)
                        elif resource_b_percentage == 0.0 and resource_a_percentage == 0.0:
                            resource_b_ratio = resource_a_ratio = 0.5
                        elif resource_b_percentage == 0.0:
                            resource_a_ratio = 1.0
                            resource_b_ratio = 0.0
                        else:
                            resource_a_ratio = 0.0
                            resource_b_ratio = 1.0
                        blended_color = tuple(map(lambda f, w: f*resource_b_ratio + w*resource_a_ratio, resource_b_color, resource_a_color))

                    rect = pygame.Rect(row * cell_size, col * cell_size, cell_size, cell_size)
                    draw_rect_alpha(screen, blended_color, rect)

            # draw agents
            mini_rect_size = 14
            for id, agent in enumerate(agents):
                if agent.is_alive():
                    x, y = agent.get_pos()
                    if enable_rendering:
                        rect = pygame.Rect(x * cell_size + (cell_size - mini_rect_size)/2, y * cell_size + (cell_size - mini_rect_size)/2, mini_rect_size, mini_rect_size)
                        pygame.draw.rect(screen, agent.get_color(), rect)

            # draw the grid
            for x in range(0, screen_width, cell_size):
                pygame.draw.line(screen, black, (x, 0), (x, screen_height))
            for y in range(0, screen_height, cell_size):
                pygame.draw.line(screen, black, (0, y), (screen_width, y))

            # update the display
            pygame.display.flip()

    clock.tick(fps)
    dt = clock.tick(fps) / 100
    time += 1




if __name__ == "__main__":
    # n_agents, agent_type, move_prob, save_to_file, run_nr, run_time, enable_rendering
    arg = (2, 'neural_agent', 1, False, 1, 1000, True)
    run_simulation(arg)