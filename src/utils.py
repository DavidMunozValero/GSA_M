import math
import numpy as np
import random

from functools import lru_cache
from scipy.spatial.distance import hamming
from typing import Mapping, Tuple


def mass_calculation(fit: np.ndarray) -> np.ndarray:
    """
    Efficiently calculates the mass of particles based on their fitness values.
    It normalizes the fitness values to compute the mass, ensuring that the sum of all masses equals 1.

    Args:
        fit (np.ndarray): Fitness values of the particles.

    Returns:
        np.ndarray: Normalized mass of the particles.
    """
    # Normalize fitness values to [0, 1] to compute mass
    f_min, f_max = fit.min(), fit.max()
    if f_max == f_min:
        return np.ones(fit.shape) / len(fit)
    else:
        normalized_fit = (fit - f_max) / (f_min - f_max)
        mass = normalized_fit / normalized_fit.sum()
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
    return g_zero * (1 - curr_iter / max_iters)


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
    return g_zero * np.exp(-alpha * curr_iter / max_iters)


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


def g_bin_field(population_size: int,
            dim: int,
            pos: np.ndarray,
            mass: np.ndarray,
            current_iter: int,
            max_iters: int,
            gravity_constant: float,
            elitist_check: int,
            r_power: int
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
        elitist_check: int : elitist check parameter
        r_power: int : power of the distance
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

    k_best = int(k_best)
    ds = sorted(range(len(mass)), key=lambda k: mass[k], reverse=True)

    force = np.zeros((population_size, dim))
    # force = Force.astype(int)

    for r in range(population_size):
        for ii in range(0, k_best):
            z = ds[ii]
            if z != r:
                x = pos[r, :]
                y = pos[z, :]
                if real:
                    esum = 0
                    for t in range(dim):
                        imval = ((x[t] - y[t]) ** 2)
                        esum = esum + imval
                    radius = math.sqrt(esum)
                else:
                    radius = hamming(x, y)

                for k in range(dim):
                    n = random.random()
                    force[r, k] = force[r, k] + n * (mass[z]) * (
                                (pos[z, k] - pos[r, k]) / (radius ** r_power + np.finfo(float).eps))

    acc = np.zeros((population_size, dim))
    for x in range(population_size):
        for y in range(dim):
            acc[x, y] = force[x, y] * gravity_constant

    return acc


def move(position: Mapping[str, np.ndarray],
         velocity: Mapping[str, np.ndarray],
         acceleration: Mapping[str, np.ndarray],
         v_max: int = 6
         ) -> Tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    """
    Updates the position and velocity of particles in the search space based on their acceleration.
    This implementation leverages vectorized operations for efficiency.

    Args:
        position (Mapping[str, np.ndarray]): Current positions of the particles.
        velocity (Mapping[str, np.ndarray]): Current velocities of the particles.
        acceleration (Mapping[str, np.ndarray]): Current accelerations of the particles.
        v_max (int): Maximum velocity of the particles.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Updated position and velocity of the particles.
    """
    # Real space
    r1 = np.random.random(position['real'].shape)  # Generate random coefficients for velocity update
    velocity['real'] = r1 * velocity['real'] + acceleration['real']  # Update velocity
    position['real'] += velocity['real']  # Update position

    # Discrete space
    r2 = np.random.random(position['discrete'].shape)  # Generate random coefficients for velocity update
    velocity['discrete'] = r2 * velocity['discrete'] + acceleration['discrete']  # Update velocity
    velocity['discrete'] = np.clip(velocity['discrete'], a_min=None, a_max=v_max)
    velocity['discrete'] = np.abs(np.tanh(velocity['discrete']))

    return position, velocity
