# -*- coding: utf-8 -*-
"""
Python code of Gravitational Search Algorithm (GSA)
Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm." 
           Information sciences 179.13 (2009): 2232-2248.	
Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/7ossam81/EvoloPy and matlab version of GSA at mathworks.

Purpose: Main file of Gravitational Search Algorithm(GSA) 
            for minimizing of the Objective Function

Code compatible:
 -- Python >= 3.9
"""
import numpy
import time

import massCalculation
import move

from gField import g_field
from gravitational_constant import g_constant, sin_chaotic_term
from solution import Solution


def GSA(objective_function,
        lower_bound: float,
        upper_bound: float,
        dim: int,
        population_size: int,
        iters: int,
        chaotic_constant=False
        ):
    """
    Main function of Gravitational Search Algorithm

    Args:
        objective_function (callable): Objective function to be minimized
        lower_bound (float): Lower bound of the search space
        upper_bound (float): Upper bound of the search space
        dim (int): Number of dimensions
        population_size (int): Number of individuals in the population
        iters (int): Maximum number of iterations
        chaotic_constant (bool): True if chaotic constant is used, False otherwise

    Returns:
        solution: Best solution obtained
    """
    # GSA parameters
    elitist_check = 1
    r_power = 1

    # Chaotic constant parameters
    w_max = 20
    w_min = 1e-10

    s = Solution()

    """ Initializations """
    vel = numpy.zeros((population_size, dim))
    fit = numpy.zeros(population_size)
    M = numpy.zeros(population_size)
    g_best = numpy.zeros(dim)
    g_best_score = float("inf")

    pos = numpy.random.uniform(low=0, high=1, size=(population_size, dim)) * (upper_bound - lower_bound) + lower_bound

    best_solution_history = []
    convergence_curve = numpy.zeros(iters)

    print("GSA is optimizing  \"" + objective_function.__name__ + "\"")

    timer_start = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(0, iters):
        for i in range(0, population_size):
            l1 = numpy.clip(pos[i, :], lower_bound, upper_bound)
            pos[i, :] = l1

            # Calculate objective function for each particle
            fitness = objective_function(l1)
            fit[i] = fitness

            if g_best_score > fitness:
                g_best_score = fitness
                g_best = l1

        """ Calculating Mass """
        M = massCalculation.massCalculation(fit, population_size, M)

        """ Calculating Gravitational Constant """
        G = g_constant(l, iters)
        if chaotic_constant:
            chValue = w_max - l * ((w_max - w_min) / iters)
            chaotic_term, _ = sin_chaotic_term(l, chValue)
            G += chaotic_term

        """ Calculating G field """
        acc = g_field(population_size, dim, pos, M, l, iters, G, elitist_check, r_power)

        """ Calculating Position """
        pos, vel = move.move(population_size, dim, pos, vel, acc)

        convergence_curve[l] = g_best_score
        best_solution_history.append(g_best)

        if l % 1 == 0:
            print(['At iteration ' + str(l + 1) + ' the best fitness is ' + str(g_best_score)])

    timer_end = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timer_end - timer_start
    s.convergence = convergence_curve
    s.solution_history = best_solution_history
    s.algorithm = "GSA"
    s.objective_function_name = objective_function.__name__

    return s
