# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point
import pickle


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def __init__(self, index, gamma = 0.95):
        ReflexCaptureAgent.__init__(self, index)
        self.pellet_consumed = False

    def get_boundary_positions(self, game_state):
        """
        Returns a list of positions along the boundary that are within your territory.
        """
        layout = game_state.get_walls()
        mid_x = layout.width // 2
        boundary_x = mid_x - 1 if self.red else mid_x  # Adjust for team color
        boundary_positions = [(boundary_x, y) for y in range(layout.height) if not layout[boundary_x][y]]

        return boundary_positions
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)  #Agent's state after taking the action
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  #self.get_score(successor)

        #Compute distance to the nearest food to punish being far from food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        #Check for nearby enemies whenever in the opponent's field
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        visible_enemies = [e for e in enemies if e.get_position() is not None and not e.is_pacman and my_state.is_pacman and e.scared_timer == 0]
        if visible_enemies:
            enemy_distances = [self.get_maze_distance(my_pos, e.get_position()) for e in visible_enemies]
            nearest_enemy_distance = min(enemy_distances)
        else:
            nearest_enemy_distance = float('inf')  #No visible enemies

        if nearest_enemy_distance <= 3:
            features['distance_to_enemy'] = 1 / (nearest_enemy_distance + 1)

        if nearest_enemy_distance <= 2 and my_pos[0]>15:
            #prioritize moving to safety
            if self.pellet_consumed == False:
                power_pellet_position = (25, 10)
            else:
                power_pellet_position = (1, 1)
            distance_to_power_pellet = self.get_maze_distance(my_pos, power_pellet_position)
            boundary_positions = self.get_boundary_positions(game_state)
            distance_to_boundary = min([self.get_maze_distance(my_pos, pos) for pos in boundary_positions])

            # Encourage movement towards the closer of either power pellet or boundary
            features['distance_to_safety'] = min(distance_to_power_pellet, distance_to_boundary)
        

        
        boundary_positions = self.get_boundary_positions(game_state)
        self.previous_position = game_state.get_agent_position(self.index)
        
        if self.previous_position in boundary_positions and my_state.num_carrying > 1 and action == Directions.WEST:
            features['crossing_bonus'] = 1
        '''boundary_positions = self.get_boundary_positions(game_state)
        dist_home = min(self.get_maze_distance(my_pos, position) for position in boundary_positions)
        if my_state.num_carrying > 2:
            features['proximity_to_home'] = 1/(1+dist_home)
            if dist_home == 0:
                features['crossing_bonus'] = 1 '''

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1


        if my_pos == (21,4) or my_pos == (23,6) or my_pos == (24,7)  or my_pos == (28,12):
            features['value_pos'] = -3
        if my_pos == (22,4) or my_pos == (24,6):
            features['value_pos'] = -1
        if my_pos == (25,10) and  not self.pellet_consumed:
            self.pellet_consumed = True
        print(action, features)
        return features  
        
    def get_weights(self, game_state, action):

        return {'successor_score': 500, 'distance_to_food': -1, 'distance_to_enemy': -100, 'distance_to_safety': -200, 'proximity_to_home': 500, 'crossing_bonus': 5000,'stop': -100, 'reverse': -5, 'value_pos':50}
        

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}