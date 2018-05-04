import random
import pygame, sys
from pygame.locals import *
from collections import defaultdict
import copy
import math

# converts continuous space to discrete
def discretize(state):

    if state == ():
        return ()

    ball_x = copy.deepcopy(state[0])
    ball_y = copy.deepcopy(state[1])
    velocity_x = copy.deepcopy(state[2])
    velocity_y = copy.deepcopy(state[3])  
    paddle_y = copy.deepcopy(state[4])

    if ball_x <= 0.08:
        state_ball_x = 0
    elif ball_x <= 0.16:
        state_ball_x = 1
    elif ball_x <= 0.25:
        state_ball_x = 2
    elif ball_x <= 0.33:
        state_ball_x = 3
    elif ball_x <= 0.41:
        state_ball_x = 4
    elif ball_x <= 0.5:
        state_ball_x = 5
    elif ball_x <= 0.58:
        state_ball_x = 6
    elif ball_x <= 0.66:
        state_ball_x = 7
    elif ball_x <= 0.75:
        state_ball_x = 8
    elif ball_x <= 0.83:
        state_ball_x = 9
    elif ball_x <= 0.91:
        state_ball_x = 10
    elif ball_x <= 1:
        state_ball_x = 11

    if ball_y <= 0.08:
        state_ball_y = 0
    elif ball_y <= 0.16:
        state_ball_y = 1
    elif ball_y <= 0.25:
        state_ball_y = 2
    elif ball_y <= 0.33:
        state_ball_y = 3
    elif ball_y <= 0.41:
        state_ball_y = 4
    elif ball_y <= 0.5:
        state_ball_y = 5
    elif ball_y <= 0.58:
        state_ball_y = 6
    elif ball_y <= 0.66:
        state_ball_y = 7
    elif ball_y <= 0.75:
        state_ball_y = 8
    elif ball_y <= 0.83:
        state_ball_y = 9
    elif ball_y <= 0.91:
        state_ball_y = 10
    elif ball_y <= 1:
        state_ball_y = 11

    if velocity_x > 0:
        state_velocity_x = 1
    else:
        state_velocity_x = -1

    if velocity_y >= 0.015:
        state_velocity_y = 1
    elif velocity_y <= -0.015:
        state_velocity_y = -1
    else:
        state_velocity_y = 0

    state_paddle_y = min(11, math.floor(12 * paddle_y / (1 - 0.2)))

    if ball_x > 1:
        state_ball_x = -1
        state_ball_y = -1
        state_velocity_x = -1
        state_velocity_y = -1
        state_paddle_y = -1

    return (state_ball_x, state_ball_y, state_velocity_x, state_velocity_y, state_paddle_y)


# given a state and action, creates next state
def time_step(current_state, best_action):

    ball_x = copy.deepcopy(current_state[0])
    ball_y = copy.deepcopy(current_state[1])
    velocity_x = copy.deepcopy(current_state[2])
    velocity_y = copy.deepcopy(current_state[3])  
    paddle_y = copy.deepcopy(current_state[4])

    # Update the paddle position based on the action chosen by agent

    paddle_y = paddle_y + (best_action * 0.04)

    if paddle_y < 0:
        paddle_y = 0
    elif paddle_y > (1 - 0.2):
        paddle_y = 1 - 0.2

    # Increment ball_x by velocity_x and ball_y by velocity_y

    ball_x = ball_x + velocity_x
    ball_y = ball_y + velocity_y

    # Bounce ball off top or bottom of screen

    if ball_y < 0:
        ball_y = -ball_y
        velocity_y = -velocity_y
    
    if ball_y > 1:
        ball_y = 2 - ball_y
        velocity_y = -velocity_y
    
    # Bounce ball off left of right of screen

    if ball_x < 0:
        ball_x = -ball_x
        velocity_x = -velocity_x
        
    if ball_x >= 1 and ball_y >= paddle_y and ball_y <= paddle_y + 0.2:
        ball_x = 2 - ball_x
        velocity_x = -velocity_x + random.uniform(-0.015, 0.015)
        velocity_y = velocity_y + random.uniform(-0.03, 0.03)
        new_reward = 1
    elif ball_x > 1:
        return (-1, -1, -1, -1, -1), -1
    else:
        new_reward = 0
 
    # impose restrictions on velocity

    if velocity_x > 0 and velocity_x < 0.03:
        velocity_x = 0.03
    elif velocity_x < 0 and velocity_x > -0.03:
        velocity_x = -0.03
    elif abs(velocity_x) > 1:
        velocity_x = 1
    if abs(velocity_y) > 1:
        velocity_y = 1

    return (ball_x, ball_y, velocity_x, velocity_y, paddle_y), new_reward


# given a state and action, creates next state
def flipped_time_step(current_state, best_action):

    ball_x = copy.deepcopy(current_state[0])
    ball_y = copy.deepcopy(current_state[1])
    velocity_x = copy.deepcopy(current_state[2])
    velocity_y = copy.deepcopy(current_state[3])  
    paddle_y = copy.deepcopy(current_state[4])

    # Update the paddle position based on the action chosen by agent
    paddle_y = paddle_y + (best_action * 0.04)

    if paddle_y < 0:
        paddle_y = 0
    elif paddle_y > (1 - 0.2):
        paddle_y = 1 - 0.2

    # Increment ball_x by velocity_x and ball_y by velocity_y

    ball_x = ball_x + velocity_x
    ball_y = ball_y + velocity_y

    # Bounce ball off top or bottom of screen

    if ball_y < 0:
        ball_y = -ball_y
        velocity_y = -velocity_y
    
    if ball_y >= 1:
        ball_y = 2 - ball_y
        velocity_y = -velocity_y
    
    # Bounce ball off left of right of screen

    if ball_x > 1:
        ball_x = 2 - ball_x
        velocity_x = -velocity_x

    if ball_x <= 0 and ball_y >= paddle_y and ball_y <= paddle_y + 0.2:
        ball_x = -ball_x
        velocity_x = -velocity_x + random.uniform(-0.015, 0.015)
        velocity_y = velocity_y + random.uniform(-0.03, 0.03)
        new_reward = 1
    elif ball_x < 0:
        return (-1, -1, -1, -1, -1,), -1
    else:
        new_reward = 0

 
    # impose restrictions on velocity

    if velocity_x > 0 and velocity_x < 0.03:
        velocity_x = 0.03
    elif velocity_x < 0 and velocity_x > -0.03:
        velocity_x = -0.03
    elif abs(velocity_x) > 1:
        velocity_x = 1
    if abs(velocity_y) > 1:
        velocity_y = 1

    return (ball_x, ball_y, velocity_x, velocity_y, paddle_y), new_reward


# given a state and action, creates next state
def player_time_step(current_state, best_action):

    global player_y

    ball_x = copy.deepcopy(current_state[0])
    ball_y = copy.deepcopy(current_state[1])
    velocity_x = copy.deepcopy(current_state[2])
    velocity_y = copy.deepcopy(current_state[3])  
    paddle_y = copy.deepcopy(current_state[4])

    # Update the paddle position based on the action chosen by agent
    paddle_y = paddle_y + (best_action * 0.04)

    if paddle_y < 0:
        paddle_y = 0
    elif paddle_y > (1 - 0.2):
        paddle_y = 1 - 0.2

    # Increment ball_x by velocity_x and ball_y by velocity_y

    ball_x = ball_x + velocity_x
    ball_y = ball_y + velocity_y

    # Bounce ball off top or bottom of screen

    if ball_y < 0:
        ball_y = -ball_y
        velocity_y = -velocity_y
    
    if ball_y >= 1:
        ball_y = 2 - ball_y
        velocity_y = -velocity_y
    
    # Bounce ball off left of right of screen

    if ball_x <= 0 and ball_y >= player_y and ball_y <= player_y + 0.2:
        ball_x = -ball_x
        velocity_x = -velocity_x + random.uniform(-0.015, 0.015)
        velocity_y = velocity_y + random.uniform(-0.03, 0.03)
    elif ball_x < 0:
        return (-1, -1, -1, -1, -1), -1
        
    if ball_x >= 1 and ball_y >= paddle_y and ball_y <= paddle_y + 0.2:
        ball_x = 2 - ball_x
        velocity_x = -velocity_x + random.uniform(-0.015, 0.015)
        velocity_y = velocity_y + random.uniform(-0.03, 0.03)
        new_reward = 1
    elif ball_x > 1:
        return (-1, -1, -1, -1, -1), -1
    else:
        new_reward = 0
 
    # impose restrictions on velocity

    if velocity_x > 0 and velocity_x < 0.03:
        velocity_x = 0.03
    elif velocity_x < 0 and velocity_x > -0.03:
        velocity_x = -0.03
    elif abs(velocity_x) > 1:
        velocity_x = 1
    if abs(velocity_y) > 1:
        velocity_y = 1

    return (ball_x, ball_y, velocity_x, velocity_y, paddle_y), new_reward



# from new state, selects an action
def Q_learn(current_state, current_reward):

    global Q_table, N_freq

    d_current_state = discretize(current_state)

    best_action = max_action(d_current_state)

    gamma = 0.9

    next_state, next_reward = time_step(current_state, best_action)
    d_next_state = discretize(next_state)

    if d_next_state == (-1, -1, -1, -1, -1):
        return d_next_state, 0

    N_freq[(d_current_state, best_action)] += 1
    alpha = 100/float(100 + N_freq[(d_current_state, best_action)])
    # TD
    Q_table[(d_current_state, best_action)] += alpha * (next_reward + gamma * max_Qval(d_next_state) - Q_table[(d_current_state, best_action)])
    # SARSA
    #Q_table[(d_current_state, best_action)] += alpha * (next_reward + (gamma * Q_table[(d_next_state, best_action)]) - Q_table[(d_current_state, best_action)])
    if next_reward > 0:
        bounce = 1
    else:
        bounce = 0

    return next_state, bounce

def max_Qval(state):

    global Q_table

    actions = [0, 1, -1]

    if state == ():
        return -1

    max_score = -100000000

    for x in actions:
        Q = Q_table[(state, x)]
        max_score = max(max_score, Q)

    return max_score


def max_action(new_state):

    global Q_table

    actions = [0, 1, -1]

    max_action = 0
    max_score = -100000000

    for x in actions:

        if N_freq[(new_state, x)] < 10:
           best_action = x
           return best_action

        if Q_table[(new_state, x)] > max_score:
            max_score = Q_table[(new_state, x)]
            best_action = x

    return best_action


def flipped_max_action(new_state):

    global Q_table

    ball_x = copy.deepcopy(new_state[0])
    ball_y = copy.deepcopy(new_state[1])
    velocity_x = copy.deepcopy(new_state[2])
    velocity_y = copy.deepcopy(new_state[3])
    paddle_y = copy.deepcopy(new_state[4])

    ball_x = 11 - ball_x
    velocity_x = -velocity_x

    state = (ball_x, ball_y, velocity_x, velocity_y, paddle_y)

    actions = [0, 1, -1]

    max_action = 0
    max_score = -100000000

    for x in actions:

        if Q_table[(state, x)] > max_score:
            max_score = Q_table[(state, x)]
            best_action = x

    return best_action


#keydown handler
def keydown(event):

    global player_vel

    if event.key == K_UP:
        player_vel = -0.04
    elif event.key == K_DOWN:
        player_vel = 0.04


#keyup handler
def keyup(event):

    global player_vel
    
    if event.key in (K_UP, K_DOWN):
        player_vel = 0



Q_table = {}  # calculates quality of state-action combination
Q_table = defaultdict(lambda: 0, Q_table)

N_freq = {}   # frequencies of state-action pairs
N_freq = defaultdict(lambda: 0, N_freq)

bounce_num = 0

# training stage
for x in range(100000):

    if x % 1000 == 0:
        print x
        print bounce_num/float(1000)
        bounce_num = 0

    state, bounce = Q_learn((0.5, 0.5, 0.03, 0.01, 0.4), 0)

    if bounce == 1:
        bounce_num += 1

    while(1):

        state, bounce = Q_learn(state, bounce)
       
        if state == (-1, -1, -1, -1, -1):
            break

        if bounce == 1:
            bounce_num += 1

pygame.init()
fps = pygame.time.Clock()

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

ACTUAL_WIDTH = 626
ACTUAL_HEIGHT = 612
WIDTH = 600
HEIGHT = 600
PAD_WIDTH = 9
BALL_RADIUS = 4

#canvas declaration
window = pygame.display.set_mode((ACTUAL_WIDTH, ACTUAL_HEIGHT), 0, 32)
pygame.display.set_caption('Hello World')

bounce_num = 0

# testing stage
for x in range(200):

    # 1.1, 1.3
    current_state = (0.5, 0.5, 0.03, 0.01, 0.4)
    gui_current_state = (0.5 * WIDTH, 0.5 * HEIGHT, 0.03 * WIDTH, 0.01 * HEIGHT, 0.4 * HEIGHT)
    # 1.2
    #current_state = (0.5, 0.5, -0.03, 0.01, 0.4)
    #gui_current_state = (0.5 * WIDTH, 0.5 * HEIGHT, -0.03 * WIDTH, 0.01 * HEIGHT, 0.4 * HEIGHT)

    player_y = 0.4
    player_vel = 0.0

    while True:

        player_y += player_vel

        if player_y < 0:
            player_y = 0
        elif player_y > 0.8:
            player_y = 0.8

        window.fill(BLACK)
        # 1.1
        pygame.draw.line(window, WHITE, [PAD_WIDTH/2, 0], [PAD_WIDTH/2, ACTUAL_HEIGHT], PAD_WIDTH)
        pygame.draw.line(window, GREEN, [ACTUAL_WIDTH - PAD_WIDTH/2, int(gui_current_state[4])], [ACTUAL_WIDTH - PAD_WIDTH/2, int(gui_current_state[4] + (0.2 * HEIGHT))], PAD_WIDTH)
        # 1.2
        #pygame.draw.line(window, WHITE, [ACTUAL_WIDTH - PAD_WIDTH/2, 0], [ACTUAL_WIDTH - PAD_WIDTH/2, ACTUAL_HEIGHT], PAD_WIDTH)
        #pygame.draw.line(window, GREEN, [PAD_WIDTH/2, int(gui_current_state[4])], [PAD_WIDTH/2, int(gui_current_state[4] + (0.2 * HEIGHT))], PAD_WIDTH)
        # 1.3
        #pygame.draw.line(window, GREEN, [PAD_WIDTH/2, player_y * HEIGHT],[PAD_WIDTH/2, player_y * HEIGHT + (0.2 * HEIGHT)], PAD_WIDTH)
        #pygame.draw.line(window, GREEN, [ACTUAL_WIDTH - PAD_WIDTH/2, int(gui_current_state[4])], [ACTUAL_WIDTH - PAD_WIDTH/2, int(gui_current_state[4] + (0.2 * HEIGHT))], PAD_WIDTH)
        
        pygame.draw.circle(window, RED, [int(gui_current_state[0]), int(gui_current_state[1])], BALL_RADIUS, 0)

        d_current_state = discretize(current_state)

        # 1.1
        best_action = max_action(d_current_state) # REPLACE LINE: best_action = your_function(current_state)
        current_state, reward = time_step(current_state, best_action)

        # 1.2a
        #best_action = flipped_max_action(d_current_state)
        #current_state, reward = flipped_time_step(current_state, best_action)

        # 1.2b
        #best_action = max_action(d_current_state)
        #current_state, reward = flipped_time_step(current_state, best_action)

        # 1.3
        #best_action = max_action(d_current_state)
        #current_state, reward = player_time_step(current_state, best_action)

        if reward == 1:
            bounce_num += 1

        gui_current_state = (current_state[0] * WIDTH, current_state[1] * HEIGHT, current_state[2] * WIDTH, current_state[3] * HEIGHT, current_state[4] * HEIGHT)

        if current_state == (-1, -1, -1, -1, -1):
            break

        for event in pygame.event.get():
            if event.type == KEYDOWN:
                keydown(event)
            elif event.type == KEYUP:
                keyup(event)
            elif event.type == QUIT:
                pygame.quit()
                sys.exit()
     
        pygame.display.update()
        #fps.tick(30)

print bounce_num / float(200)


