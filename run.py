"""
Simulation of a stochastic system where an agent tries to reach the end of a long corrdior of size L.
The player can move forward only V steps at a time.
The corridor is populated with other agents that only move forward U cells at a time.
When the player reaches another agent, there is a probability alpha of overtaking it and moving in front.
If there are other agents together, the probability of overtaking it are multiplied (independent events).
The final result is the time it takes the agent, on average, to exit the corridor.
"""
# TODO(albert): Add a corridors to look up whether the cells are empty or not. 
# TODO(albert): When moving the agents, start from the end.

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import numba

# Parameters
L = 1000  # Length of the corridor
alpha = 0.5  # Probability of overtaking another agent
N = 300  # Number of agents
Nb = 0 # Number of blockages in the corridor
MAX_STEPS = 1000  # Number of time steps
num_simulations = 1_000  # Number of simulations
V = 2  # Velocity of the player
U = 1 # Velocity of the other agents


@numba.njit
def run_simulation(N, Nb, alpha):

    # Place blocks randomly in the corridor
    b_idxs = np.sort(np.random.choice(L-1, size=Nb, replace=False) + 1)

    # Place randomly the agents in the corridor (in places where there is no blocks)
    mask = np.zeros(L) == 0
    mask[b_idxs] = False
    k_idxs = np.sort(np.random.choice(np.arange(L)[mask][1:], size=N, replace=False)+1).astype(np.int64)

    idx = 0
    step = 0
    for step in range(MAX_STEPS):

        # Move the agents 1 cell. Remove the agents that reach the end.
        # Check that there is no block in the next cell, if there is, don't move.
#        u = np.ones(len(k_idxs), np.int64) * U
#        for j, k in enumerate(k_idxs):
#            distances = b_idxs - k
#            distances = distances[distances > 0]
#            if len(distances) > 0 and min(distances) == 1:
#                u[j] = 0
        k_idxs = k_idxs + U
        k_idxs = k_idxs[k_idxs < L]

        # Compute the distance to the agents in front of the player
        distances = k_idxs - idx

        # Check if there is a traffic jam in front of the player
        j_prev = 1
        p = 0
        for j in filter(lambda a: a > 0, distances):
            if j - j_prev == 1:
                p += 1
            else:
                break
            j_prev = j

        # If there is, see if the player can overtake it
        if p > 0:
            if np.random.rand() <= alpha**p:
                next_index = idx + U + p
            else:
                next_index = idx + U
        else:
            next_index = idx + V

        if next_index >= L:
            break

        idx = next_index

    return step


if __name__ == "__main__":

    # Get the distributuin of the escape time for different values of alpha
    plt.figure()
    for alpha in np.linspace(0.2, 0.8, 8, endpoint=True):

        steps = np.array([run_simulation(N, Nb, alpha) for _ in range(num_simulations)])
        print(f"Average steps for alpha={alpha:.1f}: {np.mean(steps):.2f} ±  {np.std(steps):.2f}")

        # Plot the escape time distribution
        k = gaussian_kde(steps)
        x = np.linspace(min(steps), max(steps), 1000)
        plt.plot(x, k(x), label=rf"$\alpha={alpha:.1f}$")
        plt.xlabel("Escape time")
        plt.title(fr"Escape time distributions for different overtaking rates ($N={N}$)", loc="left")

    plt.legend()
    plt.savefig("escape_time_alpha.png", dpi=300)
    print()

    # Distributions of the escape times for different number of agents
    alpha = 0.5
    plt.figure()
    for n in np.linspace(100, 1000, 9, endpoint=False, dtype=int):

        steps = np.array([run_simulation(n, Nb, alpha) for _ in range(num_simulations)])
        print(f"Average steps for N={n}: {np.mean(steps):.2f} ±  {np.std(steps):.2f}")

        # Plot the escape time distribution
        k = gaussian_kde(steps)
        x = np.linspace(min(steps), max(steps), 1000)
        plt.plot(x, k(x), label=rf"$N={n}$")
        plt.xlabel("Escape time")
        plt.title(fr"Escape time distributions for different number of agents ($\alpha={alpha:.1f}$)", loc="left")

    plt.legend()
    plt.savefig("escape_time_N.png", dpi=300)

    plt.show()
