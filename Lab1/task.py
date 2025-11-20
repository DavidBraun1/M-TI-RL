import numpy as np

def generate_and_save_means(num_machines):
    """
    Generate random normal distributed numbers and save them to a file.
    
    Args:
        num_machines (int): Number of machines to generate means for
    """
    means = np.zeros(num_machines)
    for i in range(num_machines):
        means[i] = np.random.normal(0, 1)
        print(f"Machine {i+1} mean: {means[i]:.4f}")
    np.save("randoms", means)

def print_saved_means():
    """Print the previously saved machine means from file."""
    means = np.load("randoms.npy")
    for i, mean in enumerate(means):
        print(f"Machine {i+1} mean: {mean:.4f}")

def simulate_slot_machines(machine_rnd, num_samples):
    """
    Simulate slot machines with given means and print their average outcomes.
    
    Args:
        machine_rnd (numpy.ndarray): Mean values for each machine
        num_samples (int): Number of samples to generate per machine
    """
    for i, mean in enumerate(machine_rnd):
        samples = np.random.normal(mean, 1, num_samples)
        sample_mean = np.mean(samples)
        print(f"Machine {i+1} simulation mean: {sample_mean:.4f}")
    print()

    def slot_machine(machine_rnd, machine_nr, machine_means, machine_games):
        number = np.random.normal(machine_rnd[machine_nr], 1)
        machine_means[machine_nr, machine_games[machine_nr]] = number
        machine_games[machine_nr] =+ 1
        
    

if __name__ == "__main__":
    # Configuration
    NUM_MACHINES = 13
    NUM_SAMPLES = 10_000

    
    # Uncomment to generate new random means
    # generate_and_save_means(NUM_MACHINES)

    # Load saved means and run simulation
    machine_rnd = np.load("randoms.npy")
    # simulate_slot_machines(machine_rnd, NUM_SAMPLES)
    print_saved_means()

    # Run for finding best machine

    EPSILON = 0.35

    # init arrays
    machine_means = np.zeros((NUM_MACHINES,NUM_SAMPLES))
    machine_games = np.zeros(NUM_MACHINES)

    for i in range(NUM_SAMPLES):

        # Choose for exploration or expoitation
        mode = np.random.uniform(0, 1)

        if mode >= EPSILON:
            # exploit
            ...
        
        elif mode < EPSILON:
            # explore
            machine_nr = np.random.random_integers(1, 13)

