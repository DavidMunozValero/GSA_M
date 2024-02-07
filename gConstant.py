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
 -- Python: 2.* or 3.*
"""

import numpy as np

from functools import cache
from typing import Tuple


def g_constant(l, iters):
    alfa = 20
    G0 = 100
    Gimd = np.exp(-alfa*float(l)/iters)
    G = G0*Gimd
    return G


@cache
def compute_x(i: int) -> float:
    if i == 0:
        return 0.7  # Initial value
    else:
        # Get the previous value of x
        prev_x = compute_x(i-1)
        return 2.3 * prev_x**2 * np.sin(np.pi * prev_x)


def sin_chaotic_term(curr_iter: int, value: float) -> Tuple[float, float]:
    x = compute_x(curr_iter)
    chaotic_term = x * value
    return chaotic_term, x
