import numpy as np
import time

from .utils import g_bin_constant, g_real_constant, g_real_field, g_bin_field, mass_calculation, move, sin_chaotic_term

from typing import Any, List, Mapping, Tuple, Union


class GSA:
    """
    Gravitational Search Algorithm

    This class contains the Gravitational Search Algorithm (GSA) optimization algorithm.

    Methods:
        optimize: Method to optimize the objective function using Gravitational Search Algorithm
    """

    def __init__(self,
                 objective_function: callable,
                 r_dim: int,
                 d_dim: int,
                 boundaries: Mapping[str, List[Union[Any, Tuple[float, float]]]],
                 ) -> None:
        """
        Initialize the GSA algorithm

        Args:
            objective_function (callable): Objective function to be minimized
            r_dim (int): Number of dimensions of real variables
            d_dim (int): Number of dimensions of discrete variables
            boundaries (Mapping[str, Tuple[float, float]]): Dictionary with the lower and upper bounds for each variable type
        """
        self.objective_function = objective_function
        self.r_dim = r_dim
        assert self.r_dim == len(
            boundaries['real']), "The number of dimensions must be equal to the number of boundaries"
        self.d_dim = d_dim
        assert self.d_dim == len(
            boundaries['discrete']), "The number of dimensions must be equal to the number of boundaries"
        self.t_dim = self.r_dim + self.d_dim

        self.real_boundaries = np.array(boundaries['real'])
        self.discrete_boundaries = np.array(boundaries['discrete'])

        self.objective_function_name = None
        self.solution_history = None
        self.convergence = None
        self.start_time = None
        self.end_time = None
        self.execution_time = None

    def _get_initial_positions(self,
                               population_size: int
                               ) -> Mapping[str, np.ndarray]:
        """
        Method to get the initial positions of the individuals in the population

        Args:
            population_size (int): Number of individuals in the population

        Returns:
            Mapping[str, numpy.ndarray]: Dictionary with the initial positions of the individuals in the population
        """
        # Initialize random positions with boundaries for each individual
        pos_r = np.zeros((population_size, self.r_dim))

        for col_index in range(self.r_dim):
            rd_lb, rd_ub = self.real_boundaries[col_index]
            pos_r[:, col_index] = np.random.uniform(low=rd_lb, high=rd_ub, size=population_size)

        pos_d = np.zeros((population_size, self.d_dim))
        for col_index in range(self.d_dim):
            dd_lb, dd_ub = self.discrete_boundaries[col_index]
            pos_d[:, col_index] = np.random.uniform(low=dd_lb, high=dd_ub, size=population_size)

        return {'real': pos_r, 'discrete': pos_d}

    def optimize(self,
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
            population_size (int): Number of individuals in the population
            iters (int): Maximum number of iterations
            elitist_check (int): Elitist check parameter
            r_power (int): Power of the distance
            chaotic_constant (bool): True if chaotic constant is used, False otherwise
            w_max (float): Maximum value of the chaotic term
            w_min (float): Minimum value of the chaotic term
        """
        # Initializations
        vel_r = np.zeros((population_size, self.r_dim))
        vel_d = np.zeros((population_size, self.d_dim))
        vel = {'real': vel_r, 'discrete': vel_d}
        fit = np.zeros(population_size)
        mass = np.zeros(population_size)
        g_best = {'real': np.zeros(self.r_dim), 'discrete': np.zeros(self.d_dim)}
        g_best_score = float("inf")

        pos = self._get_initial_positions(population_size)

        best_solution_history = []
        convergence_curve = np.zeros(iters)

        print("GSA is optimizing  \"" + self.objective_function.__name__ + "\"")

        timer_start = time.time()
        self.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

        for current_iter in range(iters):
            for i in range(population_size):
                if self.r_dim > 0:
                    l1_r = np.clip(pos['real'][i, :], self.real_boundaries[:, 0], self.real_boundaries[:, 1])
                else:
                    l1_r = np.array([])
                if self.d_dim > 0:
                    l1_d = np.clip(pos['discrete'][i, :], self.discrete_boundaries[:, 0],
                                   self.discrete_boundaries[:, 1])
                else:
                    l1_d = np.array([])

                pos['real'][i, :] = l1_r
                pos['discrete'][i, :] = l1_d

                l1 = {'real': l1_r, 'discrete': l1_d}
                # Calculate objective function for each particle
                fitness = self.objective_function(l1)
                fit[i] = fitness

                if g_best_score > fitness:
                    g_best_score = fitness
                    g_best = l1

            # Calculating Mass
            mass = mass_calculation(fit)

            # Calculating Gravitational Constant
            G_real = g_real_constant(current_iter, iters)
            G_bin = g_bin_constant(current_iter, iters)

            if chaotic_constant:
                chValue = w_max - current_iter * ((w_max - w_min) / iters)
                chaotic_term, _ = sin_chaotic_term(current_iter, chValue)
                G_real += chaotic_term
                G_bin += chaotic_term

            # Calculating G field
            if self.r_dim > 0:
                acc_r = g_real_field(population_size, self.r_dim, pos['real'], mass, current_iter, iters, G_real,
                                     elitist_check, r_power)
            else:
                acc_r = np.array([])
            if self.d_dim > 0:
                acc_d = g_bin_field(population_size, self.r_dim, pos['discrete'], mass, current_iter, iters, G_bin,
                                    elitist_check, r_power)
            else:
                acc_d = np.array([])
            acc = {'real': acc_r, 'discrete': acc_d}

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
