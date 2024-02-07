# -*- coding: utf-8 -*-
"""
Python code of Gravitational Search Algorithm (GSA)
Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm." 
           Information sciences 179.13 (2009): 2232-2248.	

Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/7ossam81/EvoloPy and matlab version of GSA at mathworks.

 -- Purpose: Main File::
                Calling the Gravitational Search Algorithm(GSA) Algorithm 
                for minimizing of an Objective Function

Code compatible:
 -- Python: 2.* or 3.*
"""
import benchmarks
import csv
import numpy
import os
import time

from GSA import GSA


def selector(func_details, popSize, Iter, chaotic_constant: bool = False):
    function_name = func_details[0]
    lb = func_details[1]
    ub = func_details[2]
    dim = func_details[3]
    print(lb, ub, dim, popSize, Iter)
    print("Function: ", getattr(benchmarks, function_name))
    return GSA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter, chaotic_constant=chaotic_constant)


# Select optimizers
gsa = True  # Code by Himanshu Mittal

# Select benchmark function
F1 = True

Algorithm = [gsa]
objectivefunc = [F1]

# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
Runs = 1

# Select chaotic constant
chaotic_constant = True

# Select general parameters for all optimizers (population size, number of iterations)
PopSize = 30
iterations = 100

# Export results ?
Export = True

# ExportToFile="YourResultsAreHere.csv"
# Automaticly generated name by date and time

save_path = "data/output/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

ExportToFile = save_path + "experiment" + time.strftime("%Y-%m-%d-%H-%M-%S_") + str(chaotic_constant) + ".csv"

# Check if it works at least once
atLeastOneIteration = False

# CSV Header for the convergence
CnvgHeader = []
# SolHeader=[]

for l in range(iterations):
    CnvgHeader.append("Iter" + str(l + 1))
    # SolHeader.append("Sol_Iter"+str(l+1))

for j in range(len(objectivefunc)):
    if objectivefunc[j]:  # start experiment if an Algorithm and an objective function is selected
        for k in range(0, Runs):

            func_details = benchmarks.getFunctionDetails(j)
            x = selector(func_details, PopSize, iterations, chaotic_constant=chaotic_constant)
            if Export:
                with open(ExportToFile, 'a') as out:
                    writer = csv.writer(out, delimiter=',')
                    if not atLeastOneIteration:  # just one time to write the header of the CSV file
                        header = numpy.concatenate(
                            [["Optimizer", "objfname", "startTime", "EndTime", "ExecutionTime"], CnvgHeader])
                        print(header)
                        writer.writerow(header)
                    a = numpy.concatenate(
                        [[x.algorithm, x.objective_function_name, x.start_time, x.end_time, x.execution_time], x.convergence])
                    writer.writerow(a)
                out.close()
            atLeastOneIteration = True  # at least one experiment

if not atLeastOneIteration:  # Faild to run at least one experiment
    print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions")
