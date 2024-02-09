import numpy
import time

from .utils import g_constant, g_field, mass_calculation, move, sin_chaotic_term


class GSA:
    """
    Gravitational Search Algorithm

    This class contains the Gravitational Search Algorithm (GSA) optimization algorithm.

    Methods:
        optimize: Method to optimize the objective function using Gravitational Search Algorithm
    """
    def __init__(self,
                 objective_function,
                 lower_bound: float,
                 upper_bound: float
                 ):
        """
        Initialize the GSA algorithm

        Args:
            objective_function (callable): Objective function to be minimized
            lower_bound (float): Lower bound of the search space
            upper_bound (float): Upper bound of the search space

        Returns:
            solution: Best solution obtained
        """
        self.objective_function = objective_function
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.objective_function_name = None
        self.solution_history = None
        self.convergence = None
        self.start_time = None
        self.end_time = None
        self.execution_time = None

    def optimize(self,
                 dim: int,
                 population_size: int,
                 iters: int,
                 elitist_check: int = 1,
                 r_power: int = 1,
                 chaotic_constant: bool = False,
                 w_max: float = 20.0,
                 w_min: float = 1e-10,
                 ) -> None:
        """
        Method to optimize the objective function using Gravitational Search Algorithm

        Args:
            dim (int): Number of dimensions
            population_size (int): Number of individuals in the population
            iters (int): Maximum number of iterations
            elitist_check (int): Elitist check parameter
            r_power (int): Power of the distance
            chaotic_constant (bool): True if chaotic constant is used, False otherwise
            w_max (float): Maximum value of the chaotic term
            w_min (float): Minimum value of the chaotic term
        """
        # Initializations
        vel = numpy.zeros((population_size, dim))
        fit = numpy.zeros(population_size)
        mass = numpy.zeros(population_size)
        g_best = numpy.zeros(dim)
        g_best_score = float("inf")

        pos = numpy.random.uniform(low=0,
                                   high=1,
                                   size=(population_size, dim)) * (
                    self.upper_bound - self.lower_bound) + self.lower_bound

        best_solution_history = []
        convergence_curve = numpy.zeros(iters)

        print("GSA is optimizing  \"" + self.objective_function.__name__ + "\"")

        timer_start = time.time()
        self.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

        for current_iter in range(iters):
            for i in range(population_size):
                l1 = numpy.clip(pos[i, :], self.lower_bound, self.upper_bound)
                pos[i, :] = l1

                # Calculate objective function for each particle
                fitness = self.objective_function(l1)
                fit[i] = fitness

                if g_best_score > fitness:
                    g_best_score = fitness
                    g_best = l1

            # Calculating Mass
            mass = mass_calculation(fit)

            # Calculating Gravitational Constant
            G = g_constant(current_iter, iters)
            if chaotic_constant:
                chValue = w_max - current_iter * ((w_max - w_min) / iters)
                chaotic_term, _ = sin_chaotic_term(current_iter, chValue)
                G += chaotic_term

            # Calculating G field
            acc = g_field(population_size, dim, pos, mass, current_iter, iters, G, elitist_check, r_power)

            # Calculating Position
            pos, vel = move(pos, vel, acc)

            convergence_curve[current_iter] = g_best_score
            best_solution_history.append(g_best)

            print(['At iteration ' + str(current_iter + 1) + ' the best fitness is ' + str(g_best_score)])

        timer_end = time.time()
        self.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.execution_time = timer_end - timer_start
        self.convergence = convergence_curve
        self.solution_history = best_solution_history
        self.objective_function_name = self.objective_function.__name__
