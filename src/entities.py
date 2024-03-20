import numpy as np
import os
import random
import time

import pandas as pd

from .utils import g_bin_constant, g_real_constant, g_field, mass_calculation, sin_chaotic_term

from copy import deepcopy
from scipy.spatial.distance import euclidean, hamming
from typing import Any, List, Mapping, Tuple, Union


class Boundaries:
    """
    Boundaries

    This class contains the boundaries for the real and discrete variables.

    Attributes:
        real (List[Union[Any, Tuple[float, float]]]): List with the lower and upper bounds for each real variable
        discrete (List[Union[Any, Tuple[int, int]]]): List with the lower and upper bounds for each discrete variable
    """

    def __init__(self,
                 real: List[Union[Any, Tuple[float, float]]],
                 discrete: List[Union[Any, Tuple[int, int]]]
                 ) -> None:
        """
        Initialize the Boundaries class

        Args:
            real (List[Union[Any, Tuple[float, float]]]): List with the lower and upper bounds for each real variable
            discrete (List[Union[Any, Tuple[int, int]]]): List with the lower and upper bounds for each discrete variable
        """
        self.real = real
        self.discrete = discrete


class Solution:
    """
    Solution

    This class contains the solution of the optimization algorithm.

    Attributes:
        real (np.ndarray): Array with the real variables
        discrete (np.ndarray): Array with the discrete variables
    """
    def __init__(self, real, discrete) -> None:
        self.real = real
        self.discrete = discrete


class Velocity:
    """
    Velocity

    This class contains the velocity of the particles.

    Attributes:
        real (np.ndarray): Array with the real variables
        discrete (np.ndarray): Array with the discrete variables
    """
    def __init__(self, real, discrete) -> None:
        self.real = real
        self.discrete = discrete


class Acceleration:
    """
    Acceleration

    This class contains the acceleration acting on the particles.

    Attributes:
        real (np.ndarray): Array with the real variables
        discrete (np.ndarray): Array with the discrete variables
    """
    def __init__(self, real, discrete) -> None:
        self.real = real
        self.discrete = discrete


class GConstant:
    """
    GConstant

    This class contains the gravitational constants for the real and discrete variables.

    Attributes:
        real (float): Gravitational constant for the real variables
        discrete (float): Gravitational constant for the discrete variables
    """
    def __init__(self, real, discrete) -> None:
        self.real = real
        self.discrete = discrete


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
                 boundaries: Boundaries,
                 is_feasible: Union[callable, None] = None,
                 custom_repair: Union[None, callable] = None
                 ) -> None:
        """
        Initialize the GSA algorithm

        Args:
            objective_function (callable): Objective function to be minimized
            r_dim (int): Number of dimensions of real variables
            d_dim (int): Number of dimensions of discrete variables
            boundaries (Mapping[str, Tuple[float, float]]): Dictionary with the lower and upper bounds for each variable
            is_feasible (callable): Function to check the feasibility of the solution
        """
        self.objective_function = objective_function
        if is_feasible is None:
            self.is_feasible = lambda _: True
        self.is_feasible = is_feasible
        self.custom_repair = custom_repair
        self.r_dim = r_dim
        self.d_dim = d_dim
        self.t_dim = self.r_dim + self.d_dim

        self.boundaries = boundaries

        self.objective_function_name = self.objective_function.__name__
        self.solution_history = None
        self.accuracy_history = None
        self.convergence = None
        self.start_time = None
        self.end_time = None
        self.execution_time = None

    def _get_initial_positions(self,
                               population_size: int
    ) -> List[Solution]:
        """
        Method to get the initial positions of the individuals in the population

        Args:
            population_size (int): Number of individuals in the population

        Returns:
            Mapping[str, numpy.ndarray]: Dictionary with the initial positions of the individuals in the population
        """
        print("Initializing positions of the individuals in the population...")

        # Initialize random positions within boundaries for real-valued features
        pos_r = np.array([np.random.uniform(low=rd_lb, high=rd_ub, size=population_size)
                          for rd_lb, rd_ub in self.boundaries.real]).T

        # Initialize random positions for discrete-valued features
        pos_d = np.array([np.random.choice(a=range(dd_lb, dd_ub + 1), size=population_size)
                          for dd_lb, dd_ub in self.boundaries.discrete]).T

        population = []
        # Ensure solutions are feasible; regenerate if not
        for sol in range(population_size):
            solution = Solution(real=pos_r[sol, :], discrete=pos_d[sol, :])
            iters = 0
            while not self.is_feasible(
                    solution) and iters < 100:  # Adding a max iteration count to prevent infinite loops
                if self.r_dim > 0:
                    for col_index, (rd_lb, rd_ub) in enumerate(self.boundaries.real):
                        pos_r[sol, col_index] = np.random.uniform(low=rd_lb, high=rd_ub)
                if self.d_dim > 0:
                    for col_index, (dd_lb, dd_ub) in enumerate(self.boundaries.discrete):
                        pos_d[sol, col_index] = np.random.choice(a=range(dd_lb, dd_ub + 1))
                solution = Solution(real=pos_r[sol, :], discrete=pos_d[sol, :])
                iters += 1
            population.append(solution)

        print("Positions of the individuals in the population successfully initialized!")
        return population

    def optimize(self,
                 population_size: int,
                 iters: int,
                 r_power: int = 1,
                 elitist_check: bool = True,
                 chaotic_constant: bool = False,
                 repair_solution: bool = False,
                 initial_population: Union[None, List[Solution]] = None,
                 w_max: float = 20.0,
                 w_min: float = 1e-10,
                 ) -> pd.DataFrame:
        """
        Method to optimize the objective function using Gravitational Search Algorithm

        Args:
            population_size (int): Number of individuals in the population
            iters (int): Maximum number of iterations
            r_power (int): Power of the distance
            elitist_check (bool): Elitist check
            chaotic_constant (bool): True if chaotic constant is used, False otherwise
            repair_solution (bool): True if the solution should be repaired, False otherwise
            initial_population (Union[None, Mapping[str, np.ndarray]]): Initial population
            w_max (float): Maximum value of the chaotic term
            w_min (float): Minimum value of the chaotic term

        Returns:
            pd.DataFrame: Dataframe with the history of the optimization process
        """
        # Initializations
        vel = [Solution(np.zeros(self.r_dim), np.zeros(self.d_dim)) for _ in range(population_size)]
        fit = np.zeros(population_size)
        mass = np.zeros(population_size)

        g_best = Solution(np.zeros(self.r_dim), np.zeros(self.d_dim))
        g_best_score = float("-inf")
        best_acc = 0.0

        if initial_population is None:
            pos = self._get_initial_positions(population_size)
        else:
            pos = initial_population

        print(f"Initial population: {pos}")

        best_solution_history = []
        convergence_curve = np.zeros(iters)

        print("GSA is optimizing  \"" + self.objective_function.__name__ + "\"")

        timer_start = time.time()
        self.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

        columns = ['Iteration', 'Fitness', 'Accuracy', 'ExecutionTime', 'Discrete', 'Real']
        history = pd.DataFrame(columns=columns)

        for current_iter in range(iters):
            for i in range(population_size):
                solution = pos[i]
                # Calculate objective function for each particle
                fitness, accuracy = self.objective_function(solution)
                fit[i] = fitness

                if fitness > g_best_score:
                    g_best = deepcopy(solution)
                    g_best_score = fitness
                    g_best = solution
                    best_acc = accuracy

            history_row = [current_iter, g_best_score, best_acc, time.time() - timer_start, g_best.discrete, g_best.real]
            history.loc[len(history)] = history_row

            # Calculating Mass
            mass = mass_calculation(fit=fit)

            # Calculating Gravitational Constant
            gravity_constant = self._calculate_gravitational_constants(current_iter=current_iter,
                                                                       max_iters=iters,
                                                                       chaotic_constant=chaotic_constant,
                                                                       w_max=w_max,
                                                                       w_min=w_min)

            # Calculate Acceleration
            acc = self._calculate_acceleration(population_size=population_size,
                                               pos=pos,
                                               mass=mass,
                                               current_iter=current_iter,
                                               max_iters=iters,
                                               gravity_constant=gravity_constant,
                                               r_power=r_power,
                                               elitist_check=elitist_check)

            # Calculate Position
            pos, vel = self._move(position=pos,
                                  velocity=vel,
                                  acceleration=acc,
                                  population=population_size,
                                  v_max=6,
                                  repair_solution=repair_solution)

            convergence_curve[current_iter] = g_best_score
            best_solution_history.append(g_best)

            print(['At iteration ' + str(current_iter + 1) + ' the best fitness is ' + str(g_best_score)])

        timer_end = time.time()
        self.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.execution_time = timer_end - timer_start
        self.convergence = convergence_curve
        self.solution_history = best_solution_history

        return history

    @staticmethod
    def _calculate_gravitational_constants(current_iter: int,
                                           max_iters: int,
                                           chaotic_constant: bool,
                                           w_max: float,
                                           w_min: float
                                           ) -> GConstant:
        """
        Method to calculate the gravitational constants

        Args:
            current_iter (int): Current iteration
            max_iters (int): Maximum number of iterations
            chaotic_constant (bool): True if chaotic constant is used, False otherwise
            w_max (float): Maximum value of the chaotic term
            w_min (float): Minimum value of the chaotic term

        Returns:
            Solution: Gravitational constants for real and discrete variables
        """
        g_real = g_real_constant(current_iter, max_iters)
        g_discrete = g_bin_constant(current_iter, max_iters)

        if chaotic_constant:
            ch_value = w_max - current_iter * ((w_max - w_min) / max_iters)
            chaotic_term, _ = sin_chaotic_term(current_iter, ch_value)
            g_real += chaotic_term
            g_discrete += chaotic_term

        return GConstant(real=g_real, discrete=g_discrete)

    def _calculate_acceleration(self,
                                population_size: int,
                                pos: List[Solution],
                                mass: np.ndarray,
                                current_iter: int,
                                max_iters: int,
                                gravity_constant: GConstant,
                                r_power: int,
                                elitist_check: bool = True
                                ) -> List[Acceleration]:
        """
        Method to calculate the acceleration acting on the particles

        Args:
            population_size (int): Number of individuals in the population
            pos (Mapping[str, np.ndarray]): Current position of the particles
            mass (np.ndarray): Mass of the particles
            current_iter (int): Current iteration number
            max_iters (int): Maximum number of iterations
            gravity_constant (Mapping[str, float]): Gravitational constant for real and discrete variables
            r_power (int): Power of the distance
            elitist_check (bool): Elitist check

        Returns:
            Acceleration: Acceleration acting on the particles
        """
        acc_r = g_field(population_size=population_size,
                        dim=self.r_dim,
                        pos=np.array([p.real for p in pos], dtype=float),
                        mass=mass,
                        current_iter=current_iter,
                        max_iters=max_iters,
                        gravity_constant=gravity_constant.real,
                        r_power=r_power,
                        elitist_check=elitist_check,
                        real=True)

        acc_d = g_field(population_size=population_size,
                        dim=self.d_dim,
                        pos=np.array([p.discrete for p in pos], dtype=float),
                        mass=mass,
                        current_iter=current_iter,
                        max_iters=max_iters,
                        gravity_constant=gravity_constant.discrete,
                        r_power=r_power,
                        elitist_check=elitist_check,
                        real=False)

        acceleration = []
        for i in range(population_size):
            r_acc = acc_r[i] if self.r_dim > 0 else None
            d_acc = acc_d[i] if self.d_dim > 0 else None
            acceleration.append(Acceleration(real=r_acc, discrete=d_acc))

        return acceleration

    def _clip_positions(self,
                        solution: Solution,
                        ) -> Solution:
        """
        Clip the positions of the individuals to the boundaries of the search space

        Args:
            pos (Mapping[str, np.ndarray]): Current position of the particles

        Returns:
            Mapping[str, np.ndarray]: Clipped positions of the individuals
        """

        if self.r_dim > 0:
            l1_r = []
            for i, val in enumerate(solution.real):
                l1_r.append(np.clip(val, self.boundaries.real[i][0], self.boundaries.real[i][1]))
        else:
            l1_r = np.array([])

        if self.d_dim > 0:
            l1_d = np.clip(solution.discrete, self.boundaries.discrete[:, 0],
                           self.boundaries.discrete[:, 1]).astype(int)
        else:
            l1_d = np.array([])

        return Solution(real=np.array(l1_r, dtype=float), discrete=np.array(l1_d, dtype=float))

    def _move(self,
              position: List[Solution],
              velocity: List[Velocity],
              acceleration: List[Acceleration],
              population: int = 1,
              v_max: int = 6,
              repair_solution: bool = False
              ) -> Tuple[List[Solution], List[Velocity]]:
        """
        Updates the position and velocity of particles in the search space based on their acceleration.
        This implementation leverages vectorized operations for efficiency.

        Args:
            position (List[Solution]): Current positions of the particles.
            velocity (List[Velocity]): Current velocities of the particles.
            acceleration (List[Acceleration]): Current accelerations of the particles.
            population (int): Number of particles.
            v_max (int): Maximum velocity of the particles.
            repair_solution (bool): True if the solution should be repaired, False otherwise.

        Returns:
            Tuple[Solution, Solution]: Updated position and velocity of the particles.
        """
        for i in range(population):
            # Update real variables (if any)
            if self.r_dim > 0:
                r1 = np.random.random(position[i].real.shape)  # Generate random coefficients for velocity update
                velocity[i].real = velocity[i].real * r1 + acceleration[i].real
                position[i].real = position[i].real + velocity[i].real  # Update position

            # Update discrete variables (if any)
            if self.d_dim > 0:
                r2 = np.random.random(position[i].discrete.shape)  # Generate random coefficients for velocity update
                velocity[i].discrete = velocity[i].discrete * r2 + acceleration[i].discrete
                velocity[i].discrete = np.clip(velocity[i].discrete, a_min=None, a_max=v_max)

                discrete_move_probs = np.abs(np.tanh(velocity[i].discrete))  # Apply tanh activation function
                rand = np.random.rand(*discrete_move_probs.shape)

                position[i].discrete[rand < discrete_move_probs] = 1 - position[i].discrete[rand < discrete_move_probs]
                position[i].discrete = position[i].discrete.astype(int)

                if not np.any(position[i].discrete):
                    max_index = np.argmax(position[i].discrete)
                    position[i].discrete[max_index] = 1

            new_solution = Solution(position[i].real, discrete=position[i].discrete)

            if not self.is_feasible(new_solution):
                if repair_solution:
                    new_solution = self.custom_repair(new_solution)
                else:
                    new_solution = self._clip_positions(solution=new_solution)

            position[i] = new_solution

        return position, velocity

    @staticmethod
    def set_seed(self, seed: int) -> None:
        """
        Set seed for the random number generator.

        Args:
            seed (int): Seed for the random number generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)