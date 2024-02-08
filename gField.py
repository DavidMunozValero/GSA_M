# -*- coding: utf-8 -*-
"""
Python code of Gravitational Search Algorithm (GSA)
Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm." 
           Information sciences 179.13 (2009): 2232-2248.	

Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/7ossam81/EvoloPy and matlab version of GSA at mathworks.

Purpose: Defining the gField Function
            for calculating the Force and acceleration

Code compatible:
 -- Python >= 3.9
"""

import math
import numpy as np
import random


def g_field(population_size: int,
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

    Returns:
        np.ndarray : acceleration acting on the particles
    """
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
                esum = 0
                for t in range(dim):
                    imval = ((x[t] - y[t]) ** 2)
                    esum = esum + imval

                R = math.sqrt(esum)
                for k in range(dim):
                    n = random.random()
                    force[r, k] = force[r, k] + n * (mass[z]) * ((pos[z, k] - pos[r, k]) / (R ** r_power + np.finfo(float).eps))

    acc = np.zeros((population_size, dim))
    for x in range(population_size):
        for y in range(dim):
            acc[x, y] = force[x, y] * gravity_constant

    return acc
