# -*- coding: utf-8 -*-
"""
Python code of Gravitational Search Algorithm (GSA)
Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm." 
           Information sciences 179.13 (2009): 2232-2248.	

Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/7ossam81/EvoloPy and matlab version of GSA at mathworks.

Purpose: Defining the gConstant Function
            for calculating the Gravitational Constant

Code compatible:
 -- Python >= 3.9
"""

import numpy as np

from functools import cache
from typing import Tuple


def g_constant(curr_iter: int, max_iters: int) -> float:
    """
    Calculate the gravitational constant

    Args:
        curr_iter (int): Current iteration number
        max_iters (int): Maximum number of iterations

    Returns:
        float: Gravitational constant

    """
    alfa = 20
    g_zero = 100
    Gimd = np.exp(-alfa*float(curr_iter)/max_iters)
    G = g_zero * Gimd
    return G


@cache
def compute_x(i: int) -> float:
    """
    Compute the value of x at the i-th iteration

    Args:
        i (int): Current iteration number

    Returns:
        float: Value of x at the i-th iteration for the chaotic sinusoidal term
    """
    if i == 0:
        return 0.7  # Initial value
    else:
        # Get the previous value of x
        prev_x = compute_x(i-1)
        return 2.3 * prev_x**2 * np.sin(np.pi * prev_x)


def sin_chaotic_term(curr_iter: int, value: float) -> Tuple[float, float]:
    """
    Calculate the chaotic term using the sinusoidal chaotic map

    Args:
        curr_iter (int): Current iteration number
        value (float): Value to be multiplied with the chaotic term

    Returns:
        Tuple[float, float]: Chaotic term and the value of x at the current iteration
    """
    x = compute_x(curr_iter)
    chaotic_term = x * value
    return chaotic_term, x
