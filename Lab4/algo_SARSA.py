import numpy as np
from env import RaceEnviroment

#---------------Funktionen---------------------
def make_Q(race):
    n_actions = len(race.actions)
    coords = np.argwhere(np.isin(race.track, [1, 2, 3])) 
    n_states = np.sum(np.isin(race.track, [1, 2, 3]))

    Q = np.random.randint(-5, -1, (n_states, n_actions))
    terminal_mask = race.track[coords[:,0], coords[:,1]] == 3
    Q[terminal_mask, :] = 0
    
    return Q, coords

def make_Pi(race):
    n_states = np.sum(np.isin(race.track, [1, 2, 3]))
    n_actions = len(race.actions)

    Pi = np.random.randint(1, 9, (n_states, n_actions))
    return Pi

def choose_A(S, Q, coords, race, epsilon=0.1):
    state_index = state_to_index(S, coords)
    if np.random.rand() < epsilon:
        return np.random.choice(len(race.actions))
    else:
        return np.argmax(Q[state_index, :]) + 1

def state_to_index(S, coords):
    """S ist ein Tupel (y,x), coords ist array von allen Zuständen"""
    return np.where((coords[:,0] == S[0]) & (coords[:,1] == S[1]))[0][0]


#---------------Main---------------------
if __name__ == "__main__":
    race = RaceEnviroment("b", 5, 0, -1)

    gamma = 1
    step_size = 1
    epsilon = 0.1

    Q, coords = make_Q(race)
    Pi = make_Pi(race)
    episodes = 1

    for i_episode in range(episodes):
        S = (race.y, race.x)
        state_index = state_to_index(S, coords)
        A = choose_A(S, Q, coords, race, epsilon)
