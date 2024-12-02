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
from capture import compute_noisy_distance



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
        boundary_x = mid_x - 1 if self.red else mid_x  #Adjust for team color
        boundary_positions = [(boundary_x, y) for y in range(layout.height) if not layout[boundary_x][y]]

        return boundary_positions
    
    def get_features(self, game_state, action):
        features = util.Counter()
        remaining_time = game_state.data.timeleft
        current_state = game_state.get_agent_state(self.index)
        current_pos = current_state.get_position()
        successor = self.get_successor(game_state, action)
        successor_state = successor.get_agent_state(self.index)  #Agent's state after taking the action
        successor_pos = successor.get_agent_state(self.index).get_position()
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  #self.get_score(successor)

        #Compute distance to the nearest food to punish being far from food
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(successor_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        #Check for nearby enemies whenever in the opponent's field
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        visible_enemies = [e for e in enemies if e.get_position() is not None and not e.is_pacman and successor_state.is_pacman and e.scared_timer == 0]
        if visible_enemies:
            enemy_distances = [self.get_maze_distance(successor_pos, e.get_position()) for e in visible_enemies]
            nearest_enemy_distance = min(enemy_distances)
        else:
            nearest_enemy_distance = float('inf')  #No visible enemies

        if nearest_enemy_distance < 4:
            features['distance_to_enemy'] = 1 / (nearest_enemy_distance + 1)

        if nearest_enemy_distance <= 2 and successor_pos[0]>15 and self.red:
            #prioritize moving to safety
            if self.pellet_consumed == False:
                power_pellet_position = (25, 10)
            else:
                power_pellet_position = (1, 1)
            distance_to_power_pellet = self.get_maze_distance(successor_pos, power_pellet_position)
            boundary_positions = self.get_boundary_positions(game_state)
            distance_to_boundary = min([self.get_maze_distance(successor_pos, pos) for pos in boundary_positions])

            #Encourage movement towards the closer of either power pellet or boundary
            features['distance_to_safety'] = min(distance_to_power_pellet, distance_to_boundary)

        if nearest_enemy_distance <= 2 and successor_pos[0]<16 and not self.red:
            #prioritize moving to safety
            if self.pellet_consumed == False:
                power_pellet_position = (6, 5)
            else:
                power_pellet_position = (30,14)
            distance_to_power_pellet = self.get_maze_distance(successor_pos, power_pellet_position)
            boundary_positions = self.get_boundary_positions(game_state)
            distance_to_boundary = min([self.get_maze_distance(successor_pos, pos) for pos in boundary_positions])

            #Encourage movement towards the closer of either power pellet or boundary
            features['distance_to_safety'] = min(distance_to_power_pellet, distance_to_boundary)

        #Calculate distance to boundary
        boundary_positions = self.get_boundary_positions(game_state)
        distance_to_boundary = min([self.get_maze_distance(successor_pos, pos) for pos in boundary_positions])

        #Only apply return_bonus if agent is carrying food and progressing towards the boundary
        if self.red and (successor_state.num_carrying > 1 and successor_pos[0] > 15 and nearest_enemy_distance < 9) or (successor_state.num_carrying >= 6) or (successor_state.num_carrying > 0 and remaining_time < 120) or (game_state.get_score()+successor_state.num_carrying >= 18):
            previous_distance = getattr(self, "prev_distance_to_boundary", float('inf'))
            if distance_to_boundary < previous_distance and successor_pos != getattr(self, "prev_position", None):
                features['return_bonus'] = 200 - distance_to_boundary * 4
            else:
                features['return_bonus'] = 0  #Remove bonus if not progressing
            self.prev_distance_to_boundary = distance_to_boundary  #Track distance for comparison
            self.prev_position = successor_pos  #Track position for movement detection

        if not self.red and (successor_state.num_carrying > 1 and successor_pos[0] < 16 and nearest_enemy_distance < 9) or (successor_state.num_carrying >= 6) or (successor_state.num_carrying > 0 and remaining_time < 120) or (game_state.get_score()+successor_state.num_carrying >= 18):
            previous_distance = getattr(self, "prev_distance_to_boundary", float('inf'))
            if distance_to_boundary < previous_distance and successor_pos != getattr(self, "prev_position", None):
                features['return_bonus'] = 200 - distance_to_boundary * 4
            else:
                features['return_bonus'] = 0  #Remove bonus if not progressing
            self.prev_distance_to_boundary = distance_to_boundary  #Track distance for comparison
            self.prev_position = successor_pos  #Track position for movement detection

        if current_state.num_carrying > 0 and successor_pos[0] <= 15 and self.red:
            features['crossing_bonus'] = 5000  #Encourage crossing home
            self.prev_distance_to_boundary = float('inf')  #Reset tracking
        
        if current_state.num_carrying > 0 and successor_pos[0] >= 16 and not self.red:
            features['crossing_bonus'] = 5000  #Encourage crossing home
            self.prev_distance_to_boundary = float('inf')  #Reset tracking
        
        boundary_positions = self.get_boundary_positions(game_state)
        self.previous_position = game_state.get_agent_position(self.index)
        

        if action == Directions.STOP: features['stop'] = 1

        #Discourage entering certain positions where it's easy to get trapped when close to an enemy
        if self.red:
            if nearest_enemy_distance < 8 and any(e.scared_timer == 0 for e in enemies):
                if successor_pos == (21,4) or successor_pos == (23,6) or successor_pos == (24,7)  or successor_pos == (28,12) or successor_pos == (23,12) or successor_pos == (28,5):
                    features['value_pos'] = -5
                if successor_pos == (22,4) or successor_pos == (24,6) or successor_pos == (24,12) or successor_pos == (21,6):
                    features['value_pos'] = -4

            if nearest_enemy_distance < 6 and any(e.scared_timer == 0 for e in enemies):
                if successor_pos == (21,1) or successor_pos == (23,1) or successor_pos == (26,1) or successor_pos == (26,3) or successor_pos == (22,10) or successor_pos == (28,7) or successor_pos == (21,8):
                    features['value_pos'] = -3


            if successor_pos == (25,10) and  not self.pellet_consumed:
                self.pellet_consumed = True
        else:
            if nearest_enemy_distance < 8 and any(e.scared_timer == 0 for e in enemies):
                if successor_pos == (10,11) or successor_pos == (8,9) or successor_pos == (7,8)  or successor_pos == (3,3) or successor_pos == (8,3) or successor_pos == (3,10):
                    features['value_pos'] = -5
                if successor_pos == (9,11) or successor_pos == (7,9) or successor_pos == (7,3) or successor_pos == (10,9):
                    features['value_pos'] = -2

            if nearest_enemy_distance < 6 and any(e.scared_timer == 0 for e in enemies):
                if successor_pos == (10,14) or successor_pos == (8,14) or successor_pos == (5,14) or successor_pos == (5,12) or successor_pos == (9,5) or successor_pos == (3,8) or successor_pos == (10,7):
                    features['value_pos'] = -3


            if successor_pos == (6,5) and  not self.pellet_consumed:
                self.pellet_consumed = True
        return features  
        
    def get_weights(self, game_state, action):

        return {'successor_score': 500, 'distance_to_food': -1, 'distance_to_enemy': -100, 'distance_to_safety': -200, 'crossing_bonus': 5000,'stop': -100,'value_pos':50, 'return_bonus': 1}
        


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
        if len(invaders) > 0 and my_state.scared_timer == 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if len(invaders) > 0 and my_state.scared_timer > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = -min(dists)


        enemy_positions = [a.get_position() for a in enemies]
        invader_positions = [a.get_position() for a in enemies if a.is_pacman]
        frontier_x = 15 if self.red else 16  # Adjust based on team
        distance_to_frontier = abs(my_pos[0] - frontier_x)
        if distance_to_frontier <= 4:
            features['proximity_to_frontier'] = 1
        features['vertical_alignment'] = 0  # Default alignment feature

        # Handle visible enemies
        visible_enemy_positions = [pos for pos in enemy_positions if pos is not None]
        if visible_enemy_positions:
            dists = [self.get_maze_distance(my_pos, pos) for pos in visible_enemy_positions]
            closest_enemy_pos = visible_enemy_positions[dists.index(min(dists))]

            if abs(my_pos[0] - frontier_x) <= 3:  # Near the frontier
                # Align vertically with the closest visible enemy
                if my_pos[1] > closest_enemy_pos[1]:
                    if action == Directions.NORTH:
                        features['vertical_alignment_invader'] = -1  # Penalize moving up
                    if action == Directions.SOUTH:
                        features['vertical_alignment_invader'] = 1 #Reward moving down
                elif my_pos[1] < closest_enemy_pos[1]:
                    if action == Directions.NORTH:
                        features['vertical_alignment_invader'] = 1  # Reward moving up
                    if action == Directions.SOUTH:
                        features['vertical_alignment_invader'] = -1 #Penalize moving down

        # Handle noisy distances for unobservable enemies
        elif abs(my_pos[0] - frontier_x) <= 3:
            noisy_distances = game_state.get_agent_distances()
            if noisy_distances:
                enemy_indices = self.get_opponents(successor)
                for enemy_idx in enemy_indices:
                    if successor.get_agent_state(enemy_idx).is_pacman:
                        noisy_distance = noisy_distances[enemy_idx]
                        enemy_pos = game_state.get_agent_position(enemy_idx)

                        if enemy_pos is None:  # Enemy is out of sight
                            # Compute noisy distances to approximate vertical alignment
                            estimated_distance = compute_noisy_distance(my_pos, (frontier_x, my_pos[1]))
                            if abs(noisy_distance - estimated_distance) <= 1:  # Adjust as needed
                                # Align vertically to approximate position
                                if my_pos[1] > frontier_x:
                                    features['vertical_alignment_invader'] = -1
                                elif my_pos[1] < frontier_x:
                                    features['vertical_alignment_invader'] = 1



        if  self.red != True:
            if my_pos == (21,4) or my_pos == (23,6) or my_pos == (24,7)  or my_pos == (28,12) or my_pos == (23,12):
                features['value_pos'] = -5
            if my_pos == (22,4) or my_pos == (24,6) or my_pos == (24,12) or my_pos == (21,6):
                features['value_pos'] = -2
            if my_pos == (21,1) or my_pos == (23,1) or my_pos == (26,1) or my_pos == (26,3) or my_pos == (22,10):
                features['value_pos'] = -3
            if my_pos[0]>=29 or my_pos == (28,1) or my_pos == (28,2) or my_pos == (28,3)or my_pos == (28,4) or my_pos == (28,5):
                features['value_pos'] = -1

        else:
            if my_pos == (10,11) or my_pos == (8,9) or my_pos == (7,8)  or my_pos == (3,3) or my_pos == (8,3):
                features['value_pos'] = -5
            if my_pos == (9,11) or my_pos == (7,9) or my_pos == (7,3) or my_pos == (10,9):
                features['value_pos'] = -2
            if my_pos == (10,14) or my_pos == (8,14) or my_pos == (5,14) or my_pos == (5,12) or my_pos == (9,5):
                features['value_pos'] = -3
            if my_pos[0]<=2 or my_pos == (3,10) or my_pos == (3,11) or my_pos == (3,12)or my_pos == (3,13) or my_pos == (3,14):
                features['value_pos'] = -1
                 
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -2000, 'on_defense': 100, 'invader_distance': -100, 'stop': -100, 'reverse': -2,'proximity_to_frontier' : 100,'value_pos':5, 'vertical_alignment': 100}
