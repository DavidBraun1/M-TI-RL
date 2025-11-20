import numpy as np

# enviroment setup
grid_size = 10
num_states = grid_size**2
goal_states = [num_states - 1]
actions = ["up", "down", "right", "left"]

# discount factor
lambd = 0.8
epsi = 1e-3

# converting position and state
def pos2state(x, y):
    return y * grid_size + x 

def state2pos(state):
    x = state % grid_size
    y = state // grid_size
    return x, y

# step
def step(state, action):
    '''
    Takes a state and action and gives back the next state and reward.
    '''
    x, y = state2pos(state)

    if state in goal_states:
        return state, 0
    
    match action:
        case "up":
            y = max(y-1, 0)
        case "down":
            y = min(y+1, grid_size-1)
        case "right":
            x = min(x+1, grid_size-1)
        case "left":
            x = max(x-1, 0)
        
    next_state = pos2state(x, y)

    if next_state in goal_states:
        return next_state, 0
    else:
        return next_state, -1
    

if __name__ == "__main__":
    
    V = np.zeros(num_states)
    delta = 0
    i = 0

# Iterating throu every state till delta is small enough
    while True:
        delta = 0
        V_new = V.copy()

        for s in range(num_states):
            if s in goal_states:
                continue

            Return = 0
            for a in actions:
                 next_s, reward = step(s, a)
                 Return += 0.25 * (reward + lambd * V[next_s]) 

            V_new[s] = Return
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        i = i+1
        if delta < epsi:
            break

# extract policy
policy = np.empty(num_states, dtype=object)

for s in range(num_states):
    if s in goal_states:
        policy[s] = "x"
        continue

    best_action = None
    best_value = -1e10

    for a in actions:
        next_s, _ = step(s, a)
        q = V[next_s]

        if q > best_value:
            best_value = q
            best_action = a

    match best_action:
        case "up":
            symbol = "↑"
        case "down":
            symbol = "↓"
        case "right":
            symbol = "→"
        case "left":
            symbol = "←"
        case None:
            symbol = "?"

    policy[s] = symbol

for y in range(grid_size):
    row = []
    for x in range(grid_size):
        s = pos2state(x, y)
        row.append(policy[s])
    print(row)

print("")

for y in range(grid_size):
    row = []
    for x in range(grid_size):
        s = pos2state(x, y)
        row.append(float(V[s]))
    print(" ".join(f"{float(v):.2f}" for v in row))

print("Iterationen: ", i)