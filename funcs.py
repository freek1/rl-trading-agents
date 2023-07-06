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

def get_grid_params():
    ''' returns the size of the grid
    input: 
        None
    output:
        tuple, (width, height)
    '''
    return grid_width, grid_height, cell_size

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
    if resource == 'wood':
        return 'food'
    return 'wood'

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
    if 0 <= x + dx < g_ri_d_w_id_th and  0 <= y + dy < g_ri_d_h_ei_gh_t:
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
                if 0 <= x+dx < g_ri_d_w_id_th and 0 <= y+dy < g_ri_d_h_ei_gh_t:
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