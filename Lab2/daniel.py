epsilon = 1e-3
gamma = 0.8

import numpy as np

class GridWorld():
    def __init__(self, size=10, terminals=None, terminal_reward=0):
        self.size = size
        self.state = (0,0)
        self.actions = ['up','down','left','right']
        self.reward_per_step = -1
        self.terminal_reward = terminal_reward
        if terminals is None:
            self.terminals = [(4,4)]
        else:
            self.terminals = list(terminals)
    
    def reset(self):
        self.state = (0,0)
        return self.state

    def step(self,action):
        x,y = self.state

        if action == 'up':
            x = max(x-1,0)
        elif action == 'down':
            x = min(x+1,self.size -1)
        elif action == 'left':
            y = max(y-1,0)
        elif action == 'right':
            y = min(y+1,self.size-1)

        new_state = (x,y)
        self.state = new_state

        reward = self.reward_per_step
        return new_state, reward   
    
    def is_terminal(self,state):
        return state in self.terminals



def get_next_state(state, action, size=10):
    x, y = state
    if action == 'up':
        x = max(x - 1, 0)
    elif action == 'down':
        x = min(x + 1, size - 1)
    elif action == 'left':
        y = max(y - 1, 0)
    elif action == 'right':
        y = min(y + 1, size - 1)
    return (x, y)

def value_iteration(env,gamma,epsilon):
    size = env.size
    states = [(i,j) for i in range(size) for j in range(size)]
    actions = env.actions

    # Initial Values
    V = {s: 0 for s in states}
    iterations = 0

    while True:
        delta = 0
        V_new = V.copy()

        # Go through every state
        for s in states:
            # Check if the terminal state is reached
            if env.is_terminal(s):
                V_new[s] = 0
                continue

            action_values = []
            for a in actions:
                next_state = get_next_state(s,a,size)
                reward = env.reward_per_step
                action_values.append(reward + gamma*V[next_state])

            V_new[s] = max(action_values)
            iterations=iterations+1
            delta = max(delta, abs(V_new[s] - V[s]))
            

        V = V_new

        
        if delta < epsilon:
            break
    print("Iterations", iterations)

    return V


def extract_policy(env,V,gamma):
    size = env.size
    policy = {}
    actions = env.actions

    for s in [(i,j) for i in range(size) for j in range (size)]:
        if env.is_terminal(s):
            policy[s] = None
            continue

        best_a = None
        best_value = -float('inf')

        for a in actions:
            next_state = get_next_state(s,a,size)
            value = env.reward_per_step + gamma * V[next_state]
            if value > best_value:
                best_value = value
                best_a = a
        
        policy[s] = best_a

    return policy

env = GridWorld(size=10, terminals=[(9,9)], terminal_reward=0)
V = value_iteration(env,gamma,epsilon)
policy = extract_policy(env,V,gamma)

# Print value function
print("Value Function:")
for i in range(env.size):
    row = [f"{V[(i,j)]:6.2f}" for j in range(env.size)]
    print(" ".join(row))

# Print policy (as arrows for readability)
action_symbols = {'up':'↑','down':'↓','left':'←','right':'→', None:'T'}
print("\nOptimal Policy:")
for i in range(env.size):
    row = [action_symbols[policy[(i,j)]] for j in range(env.size)]
    print(" ".join(row))


