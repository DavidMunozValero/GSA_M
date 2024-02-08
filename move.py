# -*- coding: utf-8 -*-
"""
Python code of Gravitational Search Algorithm (GSA)
Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm." 
           Information sciences 179.13 (2009): 2232-2248.	

Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/7ossam81/EvoloPy and matlab version of GSA at mathworks.

Purpose: Defining the move Function
            for calculating the updated position

Code compatible:
 -- Python >= 3.9
"""

import numpy as np
import random

from typing import Tuple


def move(population_size: int,
         dim: int,
         position: np.ndarray,
         velocity: np.ndarray,
         acceleration: np.ndarray
         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Move the particles to the new position in the search space

    Args:
        population_size: int : population size
        dim: int : dimension of the search space
        position: np.ndarray : current position of the particles
        velocity: np.ndarray : current velocity of the particles
        acceleration: np.ndarray : current acceleration of the particles

    Returns:
        Tuple[np.ndarray, np.ndarray] : new position and velocity of the particles
    """
    for i in range(0, population_size):
        for j in range(0, dim):
            r1 = random.random()
            velocity[i, j] = r1 * velocity[i, j] + acceleration[i, j]
            position[i, j] = position[i, j] + velocity[i, j]

    return position, velocity
