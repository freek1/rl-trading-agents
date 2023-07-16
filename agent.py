import random
import numpy as np
import math

trade_threshold = 1.5
dark_brown = (60, 40, 0)
dark_green = (0, 102, 34)
pink = (255, 192, 203)
orange = (240, 70, 0)
upkeep_cost = 0.035



class Agent:
    def __init__(self, id, x, y, agent_type, color, market):
        
        self.x = x
        self.y = y
        self.id = id
        self.alive = True
        self.color = color
        self.time_alive = 0
        self.resource_a_capacity = 30
        self.resource_b_capacity = 30
        self.current_stock = {
            "resource_a": random.uniform(4, 8),
            "resource_b": random.uniform(4, 8),
        }
        self.upkeep_cost = {
            "resource_a": upkeep_cost,
            "resource_b": upkeep_cost,
        }
        self.behaviour = "explore"  # 'trade_resource_a', 'trade_resource_b'
        self.agent_type = agent_type
        self.movement = "random"  # initialize as random for all agent types, since their movement changes only when wanting to trade
        self.goal_position = (None, None)  # x, y
        self.nearest_neighbors = []  # list of (x,y) of the nearest neighbors
        self.closest_market_pos = (None, None)
        self.in_market = False
        self.market = market
        self.treshold_new_neighbours = 0
        self.utility = 0
        self.food_locations = np.zeros((28,))




    def update_behaviour(self, positions_tree, agent_positions, agents, k, view_radius):
        # update trade behaviour
        
        # ratio = self.calculate_resource_ratio("resource_a", "resource_b")
        # if (ratio > trade_threshold):
        #     self.color = dark_brown
        #     self.behaviour = "trade_resource_a"  # means selling resource_a
        #     # adapt movement behaviour
        #     self.movement = self.agent_type
        #     if len(self.nearest_neighbors) == 0 and self.treshold_new_neighbours == 0:
        #         self.get_set_closest_neighbor(positions_tree, agents, min(k, len(agent_positions)), view_radius)
        #         self.treshold_new_neighbours=50
        # elif (1 / ratio > trade_threshold):
        #     self.color = dark_green
        #     self.behaviour = "trade_resource_b"  # means selling resource_b
        #     self.movement = self.agent_type
        #     if len(self.nearest_neighbors) == 0 and self.treshold_new_neighbours == 0:
        #         self.get_set_closest_neighbor(positions_tree, agents, min(k, len(agent_positions)), view_radius)
        #         self.treshold_new_neighbours=50
        # else:
            self.color = orange
            self.behaviour = "explore"
            self.movement = "explore"


    def update_time_alive(self):
        self.time_alive += 1

    def get_pos(self):
        return self.x, self.y
    
    def choose_step(self):
        """pick the next direction to walk in for the agent
        input:
            self: agent
        output:
            dx
            dy
        """
        dx, dy = 0, 0
        
        # if self.behaviour == "explore":
        #     [up_prob, right_prob, down_prob, left_prob] = self.explore_network(self.x, self.y, self.current_stock["resource_a"], self.current_stock["resource_b"])
        # if self.behaviour == "trade_resource_a":
        #     [up_prob, right_prob, down_prob, left_prob] = self.trade_resource_a_network(self.x, self.y, self.current_stock["resource_a"], self.current_stock["resource_b"])
        # if self.behaviour == "trade_resource_b":
        #     [up_prob, right_prob, down_prob, left_prob] = self.trade_resource_b_network(self.x, self.y, self.current_stock["resource_a"], self.current_stock["resource_b"])
        
        # move = np.argmax([up_prob, right_prob, down_prob, left_prob])

        # if move == 0:
        #     dy = -1
        # elif move == 1:
        #     dx = 1
        # elif move == 2:
        #     dy = 1
        # elif move == 3:
        #     dx = -1
                
        # if self.movement == "random":
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)

        return dx, dy

    def set_in_market(self, in_market):
        self.in_market = in_market

    

    def get_time_alive(self):
        return self.time_alive

    def set_movement(self, movement):
        self.movement = movement

    def set_food_locations(self, food_locations):
        self.food_locations = food_locations

    def get_food_locations(self):
        return np.array(self.food_locations)
    
            
    def get_set_closest_neighbor(self, positions_tree, agents, k, view_radius):
        # update agent position for the k_d-tree
        x, y = self.get_pos()
        
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
                if self != agents[ids]:
                    neighboring_agents.append(agents[ids])
            self.set_nearest_neighbors(neighboring_agents)
    
    

    def compatible(self, agent_b):
        """compatible if both agents are in market when this is the simulation situation."""
        if self.agent_type == "pathfind_market":
            if self.in_market and agent_b.in_market:
                if (
                    self.behaviour == "trade_resource_a"
                    and agent_b.get_behaviour() == "trade_resource_b"
                ) or (
                    self.behaviour == "trade_resource_b"
                    and agent_b.get_behaviour() == "trade_resource_a"
                ):
                    return True
        elif self.agent_type != "pathfind_market" and (
            (self.behaviour == "trade_resource_a" and agent_b.get_behaviour() == "trade_resource_b")
            or (
                self.behaviour == "trade_resource_b"
                and agent_b.get_behaviour() == "trade_resource_a"
            )
        ):
            return True
        else:
            return False

    def trade(self, agent_b):
        traded_quantity = 0.0
        minimum_difference = min(abs(self.current_stock["resource_a"] - self.current_stock["resource_b"]), abs(agent_b.current_stock["resource_a"] - agent_b.current_stock["resource_b"]))
        traded_quantity = minimum_difference/2.0
        # sell resource_a for resource_b
        if self.behaviour == "trade_resource_a":
            self.color = pink
            agent_b.set_color = pink
            self.current_stock["resource_a"] -= traded_quantity
            agent_b.current_stock["resource_a"] += traded_quantity
            agent_b.current_stock["resource_b"] -= traded_quantity
            self.current_stock["resource_b"] += traded_quantity
        # sell resource_b for resource_a  
        elif self.behaviour == "trade_resource_b":
            self.color = pink
            agent_b.set_color = pink
            self.current_stock["resource_b"] -= traded_quantity
            agent_b.current_stock["resource_b"] += traded_quantity
            agent_b.current_stock["resource_a"] -= traded_quantity
            self.current_stock["resource_a"] += traded_quantity

        return traded_quantity

    def remove_closest_neighbor(self):
        """removes closest neighbor from list"""
        self.nearest_neighbors.pop(0)
        
    def find_non_market_square(self):
        idx_market_False = np.argwhere(np.invert(self.market))
        smallest_distance = np.inf
        x_nmp, y_nmp = 0, 0
        for x_market, y_market in idx_market_False:
            distance = math.dist([x_market, y_market], [self.x, self.y])
            if distance < smallest_distance:
                smallest_distance = distance
                x_nmp, y_nmp = x_market, y_market
        return x_nmp, y_nmp

    def add_resource_a_location(self, pos):
        if pos not in self.resource_a_locations:
            self.resource_a_locations.append(pos)

    def remove_resource_a_location(self, pos):
        if pos in self.resource_a_locations:
            self.resource_a_locations.remove(pos)

    def add_resource_b_location(self, pos):
        if pos not in self.resource_b_locations:
            self.resource_b_locations.append(pos)

    def remove_resource_b_location(self, pos):
        if pos in self.resource_b_locations:
            self.resource_b_locations.remove(pos)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def upkeep(self):
        ''' Update upkeep and update utility value'''
        for resource in self.current_stock.keys():
            self.current_stock[resource] -= self.upkeep_cost[resource]
            if self.current_stock[resource] < 0:
                self.alive = False

            # Update utility
            self.utility += self.current_stock[resource]

    def gather_resource(self, chosen_resource, gather_amount):
        self.current_stock[chosen_resource] += gather_amount

    def is_at(self, x, y):
        return self.x == x and self.y == y

    def get_behaviour(self):
        return self.behaviour

    def get_color(self):
        return self.color

    def set_color(self, color):
        self.color = color

    def is_alive(self):
        return self.alive

    def preferred_resource(self):
        if self.current_stock["resource_b"] < self.current_stock["resource_a"]:
            return "resource_b", self.calculate_resource_ratio("resource_b", "resource_a")
        else:
            return "resource_a", self.calculate_resource_ratio("resource_a", "resource_b")

    def calculate_resource_ratio(self, resource_1: str, resource_2: str):
        return self.current_stock[resource_1] / self.current_stock[resource_2]

    def get_capacity(self, chosen_resource):
        if chosen_resource == "resource_a":
            return self.resource_a_capacity
        elif chosen_resource == "resource_b":
            return self.resource_b_capacity

    def get_current_stock(self, chosen_resource):
        return self.current_stock[chosen_resource]

    def set_pos(self, x, y):
        self.x = x
        self.y = y

    def get_i_d(self):
        return self.id

    def set_nearest_neighbors(self, nearest_neighbors):
        self.nearest_neighbors = nearest_neighbors

    def set_closest_market_pos(self, closest_market_pos):
        self.closest_market_pos = closest_market_pos
        
    def get_nearest_neigbors(self):
        return self.nearest_neighbors
    
    def get_treshold_new_neighbours(self):
        return self.treshold_new_neighbours
    
    def update_treshold_new_neighbours(self):
        self.treshold_new_neighbours -= 1
