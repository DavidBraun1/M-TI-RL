import numpy as np
import pandas as pd

actions = [[1,1],[1,0],[1,-1],[0,1],[0,0],[0,-1],[-1,1],[-1,0],[-1,-1]]

# 0=No Track, 1=Track, 2=Start, 3=Finish
racetrack_b = np.array([
        [0, 1, 1, 1, 1, 1, 3],
        [0, 1, 1, 1, 1, 1, 3],
        [0, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [2, 2, 2, 0, 0, 0, 0]
            ])

class RaceEnviroment:
    def __init__(self, track, speed, reward):
        self.track = track
        self.max_speed = speed
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.reward = reward
        self.reset()

    def starting_pos(self):
        ...

    def reset(self):
        self.y = 0
        # noch für alle Tracks verallgemeinern
        # starting_pos()
        self.x = np.random.randint(0,3)
        print(f"Starting Position: {self.x+1, self.y+1}")

    def move(self, A):
        # Vars
        oob = False
        GOAL = False
        x_old = self.x
        y_old = self.y
        
        # change vx, if it is allowed
        if 5 >= self.vx + A[0] >= 0:
            self.vx = self.vx + A[0]
        # change vy, if it is allowed
        if 5 >= self.vy + A[1] >= 0:
            self.vy = self.vy + A[1]

        # change position
        self.y = self.y + self.vy
        self.x = self.x + self.vx
        possible_s = self.track[y_old:self.y, x_old:self.x]

        if np.any(possible_s == 3):
            GOAL = True

        if self.track[self.y, self.x] == 0:
            oob = True

        return GOAL, oob

    def step(self, A_idx):

        GOAL, oob = self.move(actions[A_idx-1])
        
        print(f"Current Position: ({self.x+1},{self.y+1})\n\n")

        if GOAL:
            print("Finish Line crossed!\n")
            self.reset()

        if oob and not GOAL:
            print("You went Out Of Bounds!\n")
            self.reset()

    def test_move(A):
        #Bewegung über konsole
        ...

if __name__ == "__main__":
    for i in range(1, 10):
        print(f"Action {i}: {actions[i-1]}")
    print()

    RaceTrackB = RaceEnviroment(racetrack_b, 5, -1)

    
    while True:
        action = int(input("Enter action (1-9): "))
        RaceTrackB.step(action)