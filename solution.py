# -*- coding: utf-8 -*-
"""
Python code of Gravitational Search Algorithm (GSA)
Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm." 
           Information sciences 179.13 (2009): 2232-2248.	

Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/7ossam81/EvoloPy and matlab version of GSA at mathworks.

 -- Purpose: Defining the solution class
 
Code compatible:
 -- Python: 2.* or 3.*

"""


class Solution:
    def __init__(self):
        self.best = 0
        self.best_individual = []
        self.convergence = []
        self.solution_history = []
        self.optimizer = ""
        self.objective_function_name = ""
        self.start_time = 0
        self.end_time = 0
        self.execution_time = 0
        self.lower_bound = 0
        self.upper_bound = 0
        self.dim = 0
        self.population_number = 0
        self.max_iters = 0
