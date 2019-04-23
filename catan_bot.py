
# coding: utf-8

# In[1]:


import numpy as np
from catan import Catan, CatanException, get_random_dice_arrangement, Player, Game, simulate_1p_game, simulate_1p_game_with_data
import matplotlib.pyplot as plt
from itertools import repeat
import sys
import copy


# In[2]:


def action(self):
    print("__________________________________________")        
    print("NEW ACTION!")
    starting_resources = copy.deepcopy(self.resources)
    more_actions = True
    
    print(self.points)
    
    if self.points == 0 and self.if_can_buy("settlement"):
        (x,y) = self.preComp
        if not self.board.if_can_build("settlement", x, y):
            board_scores = make_board_scores(None, self.board)
            settlement_scores = make_settlement_scores(self.board, board_scores)
            new_vertex = np.argmax(np.delete(settlement_scores, self.board.get_vertex_number(x,y)))
            (x,y) = self.board.get_vertex_location(new_vertex)
            
        loc = (x, y)
        self.buy("settlement", x, y) # we determined previously    
        print("bought first settlement at " + str(loc))   
    
    while more_actions:
        starting_resources = copy.deepcopy(self.resources)        
        (goal, loc) = determine_goal(self)
        print(goal)      

        if goal == "settlement":
            (x, y) = loc
            if self.board.get_vertex_number(x, y) in [v for road in self.get_roads() for v in road]:
                if self.if_can_buy("settlement"):    
                    self.buy("settlement", x, y)
                    print("bought settlement at " + str(loc)) 
            elif len(self.get_settlements()) > 0 and self.if_can_buy("road"): 
                    (v0, v1) = optimal_road(self, loc)
                    self.buy("road", v0, v1)
                    print("bought road " + str(v0) + str(v1))
            else:
                modified_trade(self, goal)
        elif goal == "city":
            if self.if_can_buy("city"):
                (x, y) = loc
                self.buy("city", x, y)            
                print("bought city at " + str(loc))              
            else:
                modified_trade(self, goal)
        elif goal == "card":
            if self.if_can_buy("card"):
                self.buy("card")
                print("bought card")   
            else:
                modified_trade(self, goal)

        if np.array_equal(starting_resources, self.resources):
            more_actions = False
            
    return


# In[4]:


def dumpPolicy(self, max_resources):
    print("DUMPED")
    new_resources = np.minimum(self.resources, max_resources // 3)
    return self.resources - new_resources


# In[5]:


def planBoard(baseBoard):
    board_scores = make_board_scores(None, baseBoard)
    settlement_scores = make_settlement_scores(baseBoard, board_scores)
    vertex = np.argmax(settlement_scores)
    return baseBoard.get_vertex_location(vertex)


# In[3]:


def determine_goal(self):
    costs = np.array([[2, 1, 1],
                      [1, 2, 2],
                      [0, 3, 3],
                      [1, 1, 0]])    
    
    settlement_loc = optimal_settlement(self)
    num_roads = get_closest(self, settlement_loc)[1]
    
#     buildings = [("settlement", costs[0]+num_roads*costs[3]), ("city", costs[1]), ("card", costs[2])]
    buildings = [("settlement", costs[0]+costs[3]), ("city", costs[1]), ("card", costs[2])]
    goal = "settlement"
    max_value = 0
    for building in buildings:
        if building[0] == "city" and self.get_settlements() == []:
            continue
        time = hitting_time(self, self.resources, building[1])
        usefulness = 0
        curr_value = 1/time + usefulness
        if curr_value > max_value:
            goal = building[0]
            max_value = curr_value
    
    loc = None
    
    if goal == "settlement":
        loc = settlement_loc
    elif goal == "city":
        loc = optimal_city(self)
    
    return (goal, loc)


# In[6]:


def optimal_settlement(self):

    opt_set_loc = (0, 0)
    max_score = 0
    A = 5 #A is just some weight    
    
    board_scores = make_board_scores(self, self.board)
    settlement_scores = make_settlement_scores(self.board, board_scores)

    for row in range(self.board.height + 1):
        for col in range(self.board.width + 1):
            curr_settlement = (col, row)
            if self.board.if_can_build("settlement", curr_settlement[0], curr_settlement[1]):
                dist = get_closest(self, curr_settlement)[1]            
                score = settlement_scores[row][col] + A / (dist+0.01)

                if score > max_score:
                    opt_set_loc = curr_settlement
                    max_score = score
    return opt_set_loc


# In[9]:


def optimal_city(self):

    board_scores = make_board_scores(self, self.board)
    settlement_scores = make_settlement_scores(self.board, board_scores)
    opt_s = self.preComp
    val = 0
    for settlement in self.get_settlements():
        s = self.board.get_vertex_location(settlement)
        score = settlement_scores[s[1]][s[0]]
        if score > val:
            val = score
            opt_s = s
    return opt_s


# In[8]:


def optimal_road(self, goal):
    curr_settlements = self.get_settlements()
    curr_roads = self.get_roads()
    
    closest = get_closest(self, goal)[0]
    hor = goal[0] - self.board.get_vertex_location(closest)[0]
    ver = goal[1] - self.board.get_vertex_location(closest)[1]    
    v2 = closest
    
    if abs(hor) >= abs(ver):
        if hor < 0:
            v2 -= 1
        elif hor > 0:
            v2 += 1
    else:
        if ver < 0:
            v2 -= (self.board.width + 1)
        else:
            v2 += (self.board.width + 1)
    
    return (self.board.get_vertex_location(closest), self.board.get_vertex_location(v2))


# In[7]:


def get_closest(self, goal):
    curr_settlements = self.get_settlements()
    curr_roads = self.get_roads()    
    closest = 0
    closest_dist = 500
    for s in curr_settlements:
        coordinates = self.board.get_vertex_location(s)
        dist = abs(goal[0] - coordinates[0]) + abs(goal[1] - coordinates[1])
        if dist < closest_dist:
            closest = s
            closest_dist = dist
    
    for r in curr_roads:
        for end in r:
            r_end = self.board.get_vertex_location(end)
            dist = abs(goal[0] - r_end[0]) + abs(goal[1] - r_end[1])
            if dist < closest_dist:
                closest = end
                closest_dist = dist
    
    return (closest, closest_dist)


# In[10]:


def modified_trade(self, goal):
    ports = [] #get list of ports and everything thats in it    
    costs = np.array([[2, 1, 1],
                      [1, 2, 2],
                      [0, 3, 3],
                      [1, 1, 0]])        

    for e in self.get_settlements():
        if self.board.is_port(e):
            ports.append(self.board.which_port(e))
    for e in self.get_cities():
        if self.board.is_port(e):
            ports.append(self.board.which_port(e))

    if goal == "settlement":
        goal = costs[0] + costs[3]
    elif goal == "city":
        goal = costs[1]
    else:
        goal = costs[2]
    
    trade_away = -1 
    trade_for = -1
    curr_best = hitting_time(self, self.resources, goal) #find hitting time of current state
    trade = False
    subtract = 0 

    for i in range(len(self.resources)):
        required = 4 #start off needing 4 resources for each
        if i in ports: #seeing how many required to trade
            required = 2
        if 3 in ports:
            required = min(required, 3)
        if self.resources[i] >= required: #if have enough, try combinations of trading
            for j in range(len(self.resources)): 
                if i == j:
                    continue
                potential = copy.deepcopy(self.resources)#added in to make a copy of actual resources
                potential[i] -= required #performs trade
                potential[j] += 1
                new_time = hitting_time(self, potential, goal) #calculates new hitting time
                if new_time < curr_best:
                    curr_best = new_time 
                    trade_away = i
                    trade_for = j
                    trade = True
                    subtract = required
                    
    if trade:
        self.trade(trade_away, trade_for)
        print("TRADED")
    elif max(self.resources) > 5 and min(self.resources) == 0:
        self.trade(np.argmax(self.resources), np.argmin(self.resources))
        print("TRADED")


# In[11]:


def hitting_time(self, start, goal):
    # print("------------------------------------------------------------")
    
    d = False
    
    ordered = []
    closed = set()

    #TODO: consider player id?

    q = []
    q.append(list(start))
    cnt = 0
    
    goal = list(goal)
    
    # print("start", start, "goal", goal)
    
    def already_hit(s, goal):
        return all([s[i] >= goal[i] for i in range(len(goal))])

    def flatten_state(s, goal):
        return [min(s[0], goal[0]), min(s[1], goal[1]), min(s[2], goal[2])]
    
    ### BFS ### 
    
    s_idx = {} # state to index in ordered[]
        
#     print("resources", list(zip(range(2,13), self.board.get_resources(self.player_id))))
    # print("resources")
    # for i in range(11):
    #     print(i+2, self.board.get_resources(self.player_id)[i])
    
#     With BFS, Generate an ordering of the possible states leading up the goal state
    can_hit_goal = False
    while len(q) != 0: 
#     while not all([already_hit(state, goal) for state in q]) and len(q) != 0: 
        s = q.pop(0)
        closed.add(tuple(s))
        ordered.append(s)
        
        # print("Explored", s, "in", cnt, "steps")
        
        if s == goal:
            can_hit_goal = True
            # print("We've seen the goal", goal, "in", cnt, "steps")

        if already_hit(s, goal):
            continue
        all_nxt = [[s[0] + delta[0], s[1] + delta[1], s[2] + delta[2]] for delta in self.board.get_resources(self.player_id)]
        for nxt in all_nxt:
            nxt = flatten_state(nxt, goal)
            if tuple(nxt) not in closed:
                q.append(nxt)
                closed.add(tuple(nxt))
        cnt += 1
    
    if not can_hit_goal:
        # print("Cannot hit the goal, giving up...")
        # print("------------------------------------------------------------")
        return float("inf")
    
    for idx in range(len(ordered)):
        s_idx[tuple(ordered[idx])] = idx
        
    # print(s_idx)
        
    ### MATRIX ###
 
    num_states = len(ordered)
    
    A = np.zeros((num_states, num_states))
    np.fill_diagonal(A, -1)
    
    # Figure out the indicies (in the list "ordering") for each possible "next" states from each
    # idx_nxt[0] = a list of the indices (in "ordering") that each successor state is at
    idx_nxt = [[float("inf") for _ in range(11)] for _ in range(len(ordered))]
    for i in range(num_states):
        current = ordered[i] # TODO: rename
        
        # poss_nxt_state[0] = the state we will be in if we roll a 2
        # poss_nxt_state[10] = state we will be in if we roll a 12
        poss_nxt_states = [flatten_state((current[0] + delta[0],                                          current[1] + delta[1],                                          current[2] + delta[2]), goal)                            for delta in self.board.get_resources(self.player_id)]

        # print(current, poss_nxt_states)

        # Look for each of the poss_nxt_states
        for k in range(len(poss_nxt_states)): # len should be 11 each time  
#             for j in range(0, len(ordered)):
            nxt = poss_nxt_states[k]
            j = s_idx[tuple(nxt)]
            if ordered[j] == nxt:
                idx_nxt[i][k] = j
                    
#     print(idx_nxt)

    
    # probs_trans[0] = probability of rolling a 2
    probs_trans = [.0278, .0556, .0833, .1111, .1389, .1667, .1389, .1111, .0833, .056, .0278]

    # Fill in hitting time recursion based off idx_nxt indices
    for s in range(num_states):
        for trans in range(11):
            idx = idx_nxt[s][trans]
            prob = probs_trans[trans]
            A[s][idx] += prob
            
    b = np.array([-1 for _ in range(num_states)])
        
    # Change the hitting time of the goal state to be 0. Sanity check: modify the last rows
#     for s in range(num_states):
#         if ordered[s] == goal:
#             A[s] = np.zeros((1, num_states))
#             A[s][s] = 1
#             b[s] = 0
    A[len(A) - 1][len(A) - 1] = 1
    b[len(b) - 1] = 0
    # ^ REPLACE
        
        
    # Ax = b

    # print("A", A)
    # print("b", b)
    
    x = np.linalg.solve(A, b)
    
    # print("x", x)
    
    # print("------------------------------------------------------------")
    
    return x[0] # return the hitting time from the start state


# In[12]:


RESOURCE_SCORES = {0: 4, 1: 7, 2: 6, -1: 0}  ## MAKE SURE THESE ARE RIGHT. SHOULD BE WOOD BRICK GRAIN
RESOURCE_WEIGHT = 3 # FOR TESTING
SCARCITY_WEIGHT = 2  # Larger means prioritizing getting scarce resources more
PLAYER_RESOURCE_WEIGHT = 1 # Larger prioritizes even distribution of resources less
DICE_SCORES = {2: 1, 12: 1, 3: 2, 11: 2, 4: 3, 10: 3, 5: 4, 9: 4, 6: 5, 8: 5, 7: 0}
DICE_WEIGHT = 15 # FOR TESTING


# Returns 2D array of board scores with board[i][j]
def make_board_scores(self, board):
    scoreboard = [[] for _ in range(len(board.dice))]
    res_num = {}  # Mapping of how many tiles of a resource there are on map
    for i in range(len(board.dice)):
        for j in range(len(board.dice[0])):
            resource = board.resources[i][j]
            if resource in res_num:
                res_num[resource] += 1
            else:
                res_num[resource] = 1 if resource != -1 else -1  # Make desert undesirable (value of -1)
    
    for i in range(len(board.dice)):
        for j in range(len(board.dice[0])):
            calculate_tile_score
            scoreboard[i].append(calculate_tile_score(self, board, res_num, i, j))
            
    return np.array(scoreboard)

# Calculate the score of tile i, j on the board
def calculate_tile_score(self, board, res_num, i, j):
    # Calculates using resource inherent score, scarcity of resource, and how many resources player already has of it
    resource_score = RESOURCE_SCORES[board.resources[i][j]] +                     (len(board.dice)*len(board.dice[0]) / res_num.get(board.resources[i][j])) * SCARCITY_WEIGHT
    if self:
        resource_score += 1/self.resources[board.resources[i][j]] * PLAYER_RESOURCE_WEIGHT

    dice_score = DICE_SCORES[board.dice[i][j]]
    
    return resource_score * RESOURCE_WEIGHT + dice_score * DICE_WEIGHT


def make_settlement_scores(board, board_scores):
    scoreboard = [[] for _ in range(len(board.dice) + 1)]
    
    for i in range(len(scoreboard)):
        for j in range(len(board.dice[0]) + 1):
            scoreboard[i].append(get_settlement_score(board, board_scores, i, j))
    return np.array(scoreboard)

def get_settlement_score(board, board_scores, i, j):
    PORT_WEIGHT = .5
    if i == 0 or i == len(board_scores):
        if i == len(board_scores):  # To make indices line up
            i -= 1
            
        if j == 0:
            return board_scores[i][j] + PORT_WEIGHT
        elif j == len(board_scores[0]):
            return board_scores[i][j-1] + PORT_WEIGHT
        else:
            return board_scores[i][j] + board_scores[i][j-1]
    elif j == 0 or j == len(board_scores[0]):
        if j == len(board_scores[0]): # To make indices line up
            j -= 1
            
        if i == len(board_scores):
            return board_scores[i-1][j] + PORT_WEIGHT
        else:
            return board_scores[i-1][j] + board_scores[i][j]
    elif i == len(board_scores) and j == len(board_scores[0]):
        return board_scores[i-1][j-1] + PORT_WEIGHT
    else:
        return board_scores[i][j] + board_scores[i][j-1] + board_scores[i-1][j] + board_scores[i-1][j-1]