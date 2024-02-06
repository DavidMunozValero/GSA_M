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

from typing import Tuple


def gConstant(l, iters):
    alfa = 20
    G0 = 100
    Gimd = np.exp(-alfa*float(l)/iters)
    G = G0*Gimd
    return G


def sinChaoticTerm(curr_iter: int, value: float) -> Tuple[float, float]:
    x = [0.7]
    G_terms = []
    for i in range(0, curr_iter+1):
        x.append(2.3 * x[i]**2 * np.sin(np.pi*x[i]))
        G_terms.append(x[i] * value)

    return G_terms[curr_iter], x[curr_iter]
