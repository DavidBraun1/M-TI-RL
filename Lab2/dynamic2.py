import numpy as np

lambd = 0.8
epsi = 1e-3

# enviroment
grid_len = 10
num_states = grid_len**2
goal_states = [num_states-1]
actions = ["right", "up", "left", "down"]
 
# variables
i = 0
V_old = np.zeros(num_states)

def step(state, action):
    '''
    Performs an action for a state.

    Parameter:
        state (int): state from which action is performed
        action (string): action which is performed
    Return:
        next_state (int): the next state that is reached by perfoming the action in state
        reward (int): reward for reaching or not reaching the goal 
    '''

    if state in goal_states:
        return state, 0
    
    match action:
        case "right":
            next_state = min(state+1, ((state//10+1)*10-1))
        case "up":
            next_state = state-10 if state-10 >= 0 else state
        case "left":
            next_state = max(state-1, (state//10)*10)
        case "down":
            next_state = state+10 if state+10 < 100 else state
        case _:
            next_state = 0

    if next_state in goal_states:
        return next_state, 0
    else:
        return next_state, -1

while True:
    V = V_old.copy()
    delta = 0

    # for every state
    for s in range(num_states):
        if s in goal_states:
            continue
        Return = 0
        # for every action in every state
        for a in actions:
            # for every outcome of one action in one state
            for _ in range(1):
                next_s, reward = step(s, a)
                Return += 0.25 * 1 * (reward + lambd * V_old[next_s])
                
        V[s] = Return

    delta = np.max(np.abs(V-V_old))
    V_old = V
  
    if delta < epsi:
        break

for j in range(grid_len):
        print(" ".join(f"{float(v):.2f}" for v in V[10*j:10*j+10]))

# get best policy
print_a = np.full(num_states, "X")
for s in range(num_states):
    if s in goal_states:
        continue

    best_v = -1*np.inf
    best_a = None
    for a in actions:
        next_s, _ = step(s, a)
        if V[next_s] > best_v:
            best_v = V[next_s]
            best_a = a
 
    match best_a:
        case "up":
            best_a = "↑"
        case "down":
            best_a = "↓"
        case "left":
            best_a = "←"
        case "right":
            best_a = "→"
        case None:
            best_a = "?"

    print_a[s] = best_a

for j in range(grid_len):
        print(print_a[10*j:10*j+10])