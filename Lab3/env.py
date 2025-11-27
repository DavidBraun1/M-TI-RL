import numpy as np

# [vx, vy]
actions = [[1,1],[1,0],[1,-1],[0,1],[0,0],[0,-1],[-1,1],[-1,0],[-1,-1]]

# 0=No Track, 1=Track, 2=Start, 3=Finish
racetrack_b = np.flipud(np.array([
        [0, 1, 1, 1, 1, 1, 3],
        [0, 1, 1, 1, 1, 1, 3],
        [0, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [2, 2, 2, 0, 0, 0, 0]
]))

racetrack_c = np.flipud(np.array([
        [0,0,1,1,1,1,1,1,1,3],
        [0,1,1,1,1,1,1,1,1,3],
        [1,1,1,1,1,1,1,1,1,3],
        [1,1,1,1,1,1,1,1,1,3],
        [0,1,1,1,1,1,1,0,0,0],
        [0,1,1,1,1,1,0,0,0,0],
        [0,1,1,1,1,1,0,0,0,0],
        [0,0,1,1,1,1,0,0,0,0],
        [0,0,1,1,1,1,0,0,0,0],
        [0,0,1,1,1,1,0,0,0,0],
        [0,0,1,1,1,1,0,0,0,0],
        [0,0,1,1,1,1,0,0,0,0],
        [0,0,0,1,1,1,0,0,0,0],
        [0,0,0,1,1,1,0,0,0,0],
        [0,0,0,2,2,2,0,0,0,0],
]))

racetrack_a = np.flipud(np.array([
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3],
        [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,2,2,2,2,2,2,0,0,0,0,0,0,0,0],
]))

class RaceEnviroment:
    def __init__(self, track, max_speed, min_speed, reward):
        self.track = track
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.x = 0
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.reward = reward
        #global counter
        self.i = 1
        #print possible actions
        for i in range(1, 10):
            print(f"Action {i}: {actions[i-1]}")
        print()
        self.reset()

    def starting_pos(self):
        starts = np.argwhere(self.track == 2)
        num_starts = np.shape(starts)[0]
        rand_start = np.random.randint(0, num_starts)
        self.y, self.x = map(int, starts[rand_start])
        print(f"Starting Position: {self.x+1, self.y+1}\n")

    def reset(self):
        self.vx = 0
        self.vy = 0
        self.y = 0
        self.starting_pos()
        

    def move(self, A):
        # Vars
        oob = False
        GOAL = False
        x_old = self.x
        y_old = self.y
        
        # change vx, if it is allowed
        if self.max_speed >= self.vx + A[0] >= self.min_speed:
            self.vx = self.vx + A[0]
        # change vy, if it is allowed
        if self.max_speed >= self.vy + A[1] >= self.min_speed:
            self.vy = self.vy + A[1]

        # change position
        self.y = self.y + self.vy
        self.x = self.x + self.vx
        
        H, W = self.track.shape

        if self.x >= W or self.y >= H:
            oob = True
        elif self.track[self.y, self.x] == 0:
            oob = True

        possible_s = self.track[y_old:self.y+1, x_old:self.x+1]
        if np.any(possible_s == 3):
            GOAL = True

        return GOAL, oob

    def step(self, A_idx):

        GOAL, oob = self.move(actions[A_idx-1])
        
        print(f"Current Position: ({self.x+1},{self.y+1})")

        if GOAL:
            print("--------------\nFinish Line crossed!\n--------------\n")
            self.reset()

        if oob and not GOAL:
            print("--------------\nYou went Out Of Bounds!\n--------------\n")
            self.reset()

def test_move(racetrack):
    while True:
        print(f"Move {racetrack.i}:")
        action = int(input("Enter action (1-9): "))
        racetrack.step(action)
        racetrack.i = racetrack.i+1

if __name__ == "__main__":
    RaceTrack_B = RaceEnviroment(racetrack_a, 5, 0, -1)
    test_move(RaceTrack_B)
