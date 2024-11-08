# multi_agents.py
# --------------
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
#..

from util import manhattan_distance
from game import Directions, Actions
from pacman import GhostRules
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states]
        
        "*** YOUR CODE HERE ***"
        score = successor_game_state.get_score()

        food_distances = [manhattan_distance(food, new_pos) for food in new_food.as_list()]
        #Reward being closer to the closest food dot
        if food_distances:
            score += (20/min(food_distances))

        for ghost, scared_time in zip(new_ghost_states, new_scared_times):
            ghost_distance = manhattan_distance(new_pos, ghost.get_position())
            #If a ghost is scared, reward being closer to him (we give some margin in order to not get too close when he's about to stop being scared)
            if scared_time > 3: 
                score += (400 / ghost_distance + 1) #+1 to avoid dividing by 0
            #If a ghost is not scared, penalize being closer to him
            else:
                if ghost_distance < 2:
                    score -= 1000

        #Penalize a high num of food dots remaining (reward eating them)
        score -= 500 * successor_game_state.get_num_food()

        return score

def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth) 

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generate_successor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.get_num_agents():
        Returns the total number of agents in the game

        game_state.is_win():
        Returns whether or not the game state is a winning state

        game_state.is_lose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(agent_index, depth, game_state):
            # Check for terminal state
            if game_state.is_win() or game_state.is_lose() or depth == self.depth:
                return self.evaluation_function(game_state)

            # Maximize for Pacman (agent_index 0)
            if agent_index == 0:
                return max_value(agent_index, depth, game_state)
            # Minimize for each ghost
            else:
                return min_value(agent_index, depth, game_state)

        def max_value(agent_index, depth, game_state):
            actions = game_state.get_legal_actions(agent_index)
            if not actions:
                return self.evaluation_function(game_state)

            max_score = float('-inf')
            for action in actions:
                successor = game_state.generate_successor(agent_index, action)
                max_score = max(max_score, minimax(1, depth, successor))
            return max_score

        def min_value(agent_index, depth, game_state):
            actions = game_state.get_legal_actions(agent_index)
            if not actions:
                return self.evaluation_function(game_state)

            min_score = float('inf')
            next_agent = agent_index + 1
            if next_agent == game_state.get_num_agents():  # Last ghost, go to next depth
                next_agent = 0
                depth += 1

            for action in actions:
                successor = game_state.generate_successor(agent_index, action)
                min_score = min(min_score, minimax(next_agent, depth, successor))
            return min_score

        # Root call for Pacman (agent_index 0, initial depth 0)
        best_action = None
        best_score = float('-inf')
        for action in game_state.get_legal_actions(0):  # Pacman moves first
            successor = game_state.generate_successor(0, action)
            score = minimax(1, 0, successor)  # Start minimax with ghost layer at depth 0
            if score > best_score:
                best_score = score
                best_action = action

        return best_action
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(agent_index, depth, game_state, alpha, beta):
            # Check for terminal state
            if game_state.is_win() or game_state.is_lose() or depth == self.depth:
                return self.evaluation_function(game_state)

            # Maximize for Pacman (agent_index 0)
            if agent_index == 0:
                return max_value(agent_index, depth, game_state, alpha, beta)
            # Minimize for each ghost
            else:
                return min_value(agent_index, depth, game_state, alpha, beta)

        def max_value(agent_index, depth, game_state, alpha, beta):
            actions = game_state.get_legal_actions(agent_index)
            if not actions:
                return self.evaluation_function(game_state)

            max_score = float('-inf')
            for action in actions:
                successor = game_state.generate_successor(agent_index, action)
                max_score = max(max_score, alphabeta(1, depth, successor, alpha, beta))
                
                # Prune only if strictly greater than beta
                if max_score > beta:
                    return max_score
                alpha = max(alpha, max_score)
                
            return max_score

        def min_value(agent_index, depth, game_state, alpha, beta):
            actions = game_state.get_legal_actions(agent_index)
            if not actions:
                return self.evaluation_function(game_state)

            min_score = float('inf')
            next_agent = agent_index + 1
            if next_agent == game_state.get_num_agents():  # Last ghost, go to next depth
                next_agent = 0
                depth += 1

            for action in actions:
                successor = game_state.generate_successor(agent_index, action)
                min_score = min(min_score, alphabeta(next_agent, depth, successor, alpha, beta))
                
                # Prune only if strictly less than alpha
                if min_score < alpha:
                    return min_score
                beta = min(beta, min_score)
                
            return min_score

        # Root call for Pacman (agent_index 0, initial depth 0, alpha=-inf, beta=+inf)
        best_action = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in game_state.get_legal_actions(0):  # Pacman moves first
            successor = game_state.generate_successor(0, action)
            score = alphabeta(1, 0, successor, alpha, beta)  # Start alpha-beta with ghost layer at depth 0
            if score > best_score:
                best_score = score
                best_action = action

            # Update alpha after the root level move
            alpha = max(alpha, best_score)

        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

def better_evaluation_function(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()
    


# Abbreviation
better = better_evaluation_function
