import numpy as np

# -------------------------
# Environment setup
# -------------------------
grid_size = 10
num_states = grid_size**2
goal_state = num_states - 1
actions = ["up", "down", "right", "left"]

# Discount (λ) und Toleranz (ε)
gamma = 0.8       # λ im Pseudocode
epsi = 1e-3       # ε im Pseudocode

# -------------------------
# Helper-Funktionen
# -------------------------
def pos2state(x, y):
    return y * grid_size + x

def state2pos(state):
    return state % grid_size, state // grid_size

# Policy π(a|s):
# Beispiel: immer nach rechts, außer am Rand → dann nach unten
def policy(s):
    if s == goal_state:
        return None  # Endzustand → keine Aktion
    x, y = state2pos(s)
    if x < grid_size - 1:
        return "right"
    return "down"

# Environment Dynamics
def step(state, action):
    x, y = state2pos(state)

    if state == goal_state:
        return state, 0, True

    if action == "up":
        y = max(0, y - 1)
    elif action == "down":
        y = min(grid_size - 1, y + 1)
    elif action == "right":
        x = min(grid_size - 1, x + 1)
    elif action == "left":
        x = max(0, x - 1)

    next_state = pos2state(x, y)

    if next_state == goal_state:
        return next_state, 0, True
    else:
        return next_state, -1, False  # -1 cost every step

# -------------------------
# Policy Evaluation gemäß Pseudocode
# -------------------------
if __name__ == "__main__":
    V = np.zeros(num_states)    # v0(s) = 0
    i = 0                       # iteration counter

    while True:
        delta = 0
        V_new = V.copy()

        # Für alle Zustände s ∈ S
        for s in range(num_states):
            if s == goal_state:
                continue

            # Return ← 0
            Return = 0

            # Für alle Aktionen a ∈ A | π(a|s) > 0
            a = policy(s)
            if a is None:
                V_new[s] = 0
            else:
                ns, r, _ = step(s, a)
                # Return = π(a|s) * P(s'|s,a) * (r + λ * V(s'))
                # deterministisch → π=1, P=1
                Return = r + gamma * V[ns]
                V_new[s] = Return

            # max(|Vi+1(s) − Vi(s)|)
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        i += 1

        if delta < epsi:
            break

# -------------------------
# Ausgabe
# -------------------------
print("Value Function (V):")
for y in range(grid_size):
    row = []
    for x in range(grid_size):
        s = pos2state(x, y)
        row.append(round(V[s], 2))
    print(row)

print("\nPolicy (nur Richtung):")
for y in range(grid_size):
    row = []
    for x in range(grid_size):
        s = pos2state(x, y)
        if policy(s) is None:
            row.append("GOAL")
        else:
            row.append(policy(s))
    print(row)
