import pygame
from agent import Agent
import numpy as np
import math
import random
from sklearn.neighbors import KDTree
import copy

# set up the grid
cell_size = 20
grid_width = 40 
grid_height = 40

white = (255, 255, 255)
yellow = (255, 255, 0)
blue = (30, 70, 250)
dark_green = (0, 200, 0)
black = (0, 0, 0)

def get_move(list_of_moves):
    list_moves = []

    for moves in list_of_moves:
        dx = 0
        dy = 0 
        move = np.argmax(moves)

        # up right down left
        if move == 0:
            dy = -1
        elif move == 1:
            dx = 1
        elif move == 2:
            dy = 1
        elif move == 3:
            dx = -1

        list_moves.append([dx, dy])
    return list_moves

def setup_sim(arg):
    seed = 57
    n_agents, agent_type, move_prob, save_to_file, run_nr, run_time, enable_rendering = arg

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
    market_size = 3
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
                resource_a[x][y] = max_resource_a 
    for x in range(0, grid_height):
        for y in range(32, grid_height):
            if not market[x][y]:
                resource_a_cell_count += 1
                resource_b[x][y] = max_resource_b 

    # blob_size = 3
    # random_array = np.random.rand(20, 20)
    # blob_types = np.where(random_array < 0.5, 0, 1)
    # for x in range(0, grid_width):
    #     for y in range(0, grid_height):
    #         if (int(x / blob_size) % 2 == 0 and int(y / blob_size) % 2 == 0):
    #             x_blob_index = int(x / blob_size)
    #             y_blob_index = int(y / blob_size)
    #             if blob_types[x_blob_index, y_blob_index] == 0:
    #                 resource_a_cell_count += 1
    #                 resource_a[x][y] = max_resource_a 
    #             else:
    #                 resource_b_cell_count += 1
    #                 resource_b[x][y] = max_resource_b 

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

    market = np.full((grid_height, grid_width), False, dtype=bool)

    # creating agents
    for i in range(n_agents):
        random.seed(seed)
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

    if not enable_rendering:
        screen = 0
    args = enable_rendering, agents, agent_positions, resources, gather_amount, market, move_prob, alive_times, resource_a_regen_rate, resource_b_regen_rate, max_resources, screen, resource_a, resource_b, initial_resource_a_qty_cell, initial_resource_b_qty_cell, screen_width, screen_height, clock, fps, positions_tree, time
    
    return args

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
    for y in range(grid_height):
        for x in range(grid_width):
            if resources[resource][x][y] >= 1:
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