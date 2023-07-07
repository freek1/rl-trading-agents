import pygame
from agent import Agent
import numpy as np
import math
import random
from sklearn.neighbors import KDTree

# set up the grid
cell_size = 20
grid_width = 40 
grid_height = 40

white = (255, 255, 255)
yellow = (255, 255, 0)
blue = (30, 70, 250)
dark_green = (0, 200, 0)
black = (0, 0, 0)


def get_grid_params():
    ''' returns the size of the grid
    input: 
        None
    output:
        tuple, (width, height)
    '''
    return grid_width, grid_height, cell_size

def update_screen_resources(screen, resource_a, resource_b, market, initial_resource_a_qty_cell, initial_resource_b_qty_cell):
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
                # resource_b: green
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

def update_screen_agents(screen, agents):
    # draw agents
    mini_rect_size = 14
    for id, agent in enumerate(agents):
        if agent.is_alive():
            x, y = agent.get_pos()
            rect = pygame.Rect(x * cell_size + (cell_size - mini_rect_size)/2, y * cell_size + (cell_size - mini_rect_size)/2, mini_rect_size, mini_rect_size)
            pygame.draw.rect(screen, agent.get_color(), rect)

def update_screen_grid(screen, screen_width, screen_height):
    # draw the grid
    for x in range(0, screen_width, cell_size):
        pygame.draw.line(screen, black, (x, 0), (x, screen_height))
    for y in range(0, screen_height, cell_size):
        pygame.draw.line(screen, black, (0, y), (screen_width, y))

def draw_rect_alpha(surface, color, rect):
    ''' draws a rectangle with an alpha channel
    input: 
        surface: object
        color: tuple, r_gb
        rect: tuple, (x, y, w, h)
    output:
        None
    '''
    shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
    pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
    surface.blit(shape_surf, rect)

def choose_resource(agent:Agent, resources, gather_amount):
    ''' returns resource based on which resource is available
    input: 
        agent: object
        resources: dict
    output:
        chosen_resource: string, name of chosen resource
    '''
    x, y = agent.get_pos()
    preferred, ratio = agent.preferred_resource()
    if resources[preferred][x][y] >= gather_amount:
        return preferred
    elif resources[other_resource(preferred)][x][y] >= gather_amount:
        return other_resource(preferred)
    elif resources[preferred][x][y] < resources[other_resource(preferred)][x][y]*ratio:
        return other_resource(preferred)
    return preferred

def other_resource(resource: str):
    # return the opposing resource name
    if resource == 'resource_a':
        return 'resource_b'
    return 'resource_a'

def take_resource(agent: Agent, chosen_resource, resources, gather_amount):
    ''' takes a resource from the chosen resource
    input: 
        agent: object
        chosen_resource: string, name of chosen resource
        resources: dict
    output:
        None
    '''
    x, y = agent.get_pos()
    gathered = min(resources[chosen_resource][x][y], gather_amount)
    resources[chosen_resource][x][y] -= gathered
    agent.gather_resource(chosen_resource, gathered) 


def able_to_take_resource(agent, chosen_resource, resources):
    ''' checks if the agent is able to take a resource
    input: 
        agent: object
        chosen_resource: string, name of chosen resource
        resources: dict
    output:
        bool, True if able to take resource, False if not
    '''
    if chosen_resource == None:
        return False
    x, y = agent.get_pos()
    return agent.get_capacity(chosen_resource) > agent.get_current_stock(chosen_resource) and resources[chosen_resource][x][y] >= 1

def find_nearest_resource(agent, resource, resources):
    x_agent, y_agent = agent.get_pos()
    closest_loc = (-np.inf, -np.inf)
    closest_dist = np.inf
    for y in range(g_ri_d_h_ei_gh_t):
        for x in range(g_ri_d_w_id_th):
            if resources[resource][x][y]>=1:
                if math.dist((x_agent, y_agent), (x, y)) < closest_dist:
                    closest_dist = math.dist((x_agent, y_agent), (x, y))
                    closest_loc = x, y
    return closest_loc

def cell_available(x, y, agents):
    """
    returns True and agent if occupied
    """
    for agent in agents:
        if agent.is_at(x, y):
            return (False, agent)
    return (True, None)

def move_agent(preferred_direction, agent, agents):
    # move agent to preferred direction if possible, otherwise move randomly
    x, y = agent.get_pos()
    dx, dy = preferred_direction
    # check if preffered direction is possible 
    if 0 <= x + dx < grid_width and  0 <= y + dy < grid_height:
        new_x = x + dx
        new_y = y + dy
        if cell_available(new_x, new_y, agents)[0]:
            agent.move(dx, dy)
        else:
            found = False # available grid cell found
            possible_moves = [(dx, dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1]]
            possible_moves.remove((0,0))
            while not found and possible_moves:
                dx,dy = random.choice(possible_moves)
                possible_moves.remove((dx, dy))
                if 0 <= x+dx < grid_width and 0 <= y+dy < grid_height:
                    new_x = x + dx
                    new_y = y + dy
                    if cell_available(new_x, new_y, agents)[0]:
                        agent.move(dx, dy)
                        found = True

def find_closest_market_pos(agent: Agent, market):
    x, y = agent.get_pos()
    idx_market_True = np.argwhere(market)
    smallest_distance = np.inf
    x_cmp, y_cmp = 0, 0
    for x_market, y_market in idx_market_True:
        distance = math.dist([x_market, y_market], [x, y])
        if distance < smallest_distance:
            smallest_distance = distance
            x_cmp, y_cmp = x_market, y_market
    return x_cmp, y_cmp

def in_market(agent: Agent, market):
    x, y = agent.get_pos()
    return market[x][y]

def get_set_closest_neighbor(positions_tree, agents, agent:Agent, k, view_radius):
    # update agent position for the k_d-tree
    x, y = agent.get_pos()
    
    # distance and indices of 5 nearest neighbors within view radius
    view_radius = 20
    dist, idx = positions_tree.query([[x, y]], k=k)
    for i, d in enumerate(dist[0]):
        if d > view_radius:
            # neighbors_too_far += 1
            np.delete(dist, i)
            np.delete(idx, i)
    if len(idx) > 0:
        idx = idx[0]
        neighboring_agents = []
        for ids in idx:
            if agent != agents[ids]:
                neighboring_agents.append(agents[ids])
        agent.set_nearest_neighbors(neighboring_agents)