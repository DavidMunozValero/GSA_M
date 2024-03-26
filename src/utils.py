import math
import numpy as np
import random

from functools import lru_cache
from scipy.spatial.distance import euclidean, hamming
from typing import Tuple


def mass_calculation(fit: np.ndarray) -> np.ndarray:
    """
    Efficiently calculates the mass of particles based on their fitness values.
    It normalizes the fitness values to compute the mass, ensuring that the sum of all masses equals 1.

    Args:
        fit (np.ndarray): Fitness values of the particles.

    Returns:
        np.ndarray: Normalized mass of the particles.
    """
    f_min, f_max = fit.min(), fit.max()
    if f_max == f_min:
        # Same fit value for all particles, sum of masses will be 1
        return np.ones(fit.shape) / len(fit)

    # Normalize the fitness values and compute the mass
    normalized_fit = (fit - f_min) / (f_max - f_min)  # Fitness values are normalized to [0, 1]
    mass = normalized_fit / normalized_fit.sum()  # Mass is normalized to sum to 1
    return mass


def g_bin_constant(curr_iter: int, max_iters: int, g_zero: float = 1) -> float:
    """
    Calculates the gravitational constant at the current iteration, which decays exponentially over iterations.

    Args:
        curr_iter (int): Current iteration number.
        max_iters (int): Maximum number of iterations.
        g_zero (float): Initial value of the gravitational constant.

    Returns:
        float: Gravitational constant for the current iteration.
    """
    return g_zero * (1 - (curr_iter / max_iters))


def g_real_constant(curr_iter: int,
                    max_iters: int,
                    alpha: float = 20,
                    g_zero: float = 100
                    ) -> float:
    """
    Calculates the gravitational constant at the current iteration, which decays exponentially over iterations.

    Args:
        curr_iter (int): Current iteration number.
        max_iters (int): Maximum number of iterations.
        alpha (float): Decay rate of the gravitational constant.
        g_zero (float): Initial value of the gravitational constant.

    Returns:
        float: Gravitational constant for the current iteration.
    """
    return g_zero * np.exp(- ((alpha * curr_iter) / max_iters))


@lru_cache(maxsize=None)
def compute_x(i: int) -> float:
    """
    Computes the value of x at the i-th iteration for the chaotic sinusoidal term using recursion and caching for efficiency.

    Args:
        i (int): Iteration index.

    Returns:
        float: Value of x at the i-th iteration.
    """
    if i == 0:
        return 0.7  # Initial value
    prev_x = compute_x(i - 1)
    return 2.3 * prev_x ** 2 * np.sin(np.pi * prev_x)


def sin_chaotic_term(curr_iter: int, value: float) -> Tuple[float, float]:
    """
    Calculates the chaotic term using a sinusoidal chaotic map and multiplies it with a given value.

    Args:
        curr_iter (int): Current iteration number.
        value (float): Value to be multiplied with the chaotic term.

    Returns:
        Tuple[float, float]: Chaotic term and the value of x at the current iteration.
    """
    x = compute_x(curr_iter)
    return x * value, x


def g_field(population_size: int,
            dim: int,
            pos: np.ndarray,
            mass: np.ndarray,
            current_iter: int,
            max_iters: int,
            gravity_constant: float,
            r_power: int,
            elitist_check: bool = False,
            real: bool = True
            ) -> np.ndarray:
    """
    Calculate the force and acceleration acting on the particles

    Args:
        population_size: int : population size
        dim: int : dimension of the search space
        pos: np.ndarray : current position of the particles
        mass: np.ndarray : mass of the particles
        current_iter: int : current iteration number
        max_iters: int : maximum number of iterations
        gravity_constant: float : gravitational constant
        r_power: int : power of the distance
        elitist_check: int : elitist check parameter
        real: bool : True if the search space is real, False otherwise (discrete)

    Returns:
        np.ndarray : acceleration acting on the particles
    """
    if not dim > 0:
        return np.array([])

    final_per = 2
    if elitist_check == 1:
        k_best = final_per + (1 - current_iter / max_iters) * (100 - final_per)
        k_best = round(population_size * k_best / 100)
    else:
        k_best = population_size

    # Index of the particles sorted by their mass (descending order)
    ds = sorted(range(len(mass)), key=lambda k: mass[k], reverse=True)

    acc = np.zeros((population_size, dim))

    for r in range(population_size):
        for ii in range(k_best):
            z = ds[ii]
            if z != r:
                x = pos[r, :]
                y = pos[z, :]
                if real:
                    radius = euclidean(x, y)
                else:
                    radius = hamming(x, y)

                for k in range(dim):
                    n = random.random()
                    acc[r, k] += n * gravity_constant * (mass[z] / (radius + np.finfo(float).eps)) * (pos[z, k] - pos[r, k])

    return acc
