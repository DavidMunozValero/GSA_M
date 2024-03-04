import numpy as np
import time

import pandas as pd

from .utils import g_bin_constant, g_real_constant, g_field, mass_calculation, sin_chaotic_term

from scipy.spatial.distance import euclidean, hamming
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
                 is_feasible: Union[callable, None] = None,
                 ) -> None:
        """
        Initialize the GSA algorithm

        Args:
            objective_function (callable): Objective function to be minimized
            is_feasible (callable): Function to check the feasibility of the solution
            r_dim (int): Number of dimensions of real variables
            d_dim (int): Number of dimensions of discrete variables
            boundaries (Mapping[str, Tuple[float, float]]): Dictionary with the lower and upper bounds for each variable
        """
        self.objective_function = objective_function
        if is_feasible is None:
            self.is_feasible = lambda _: True
        self.is_feasible = is_feasible
        self.r_dim = r_dim
        self.d_dim = d_dim
        self.t_dim = self.r_dim + self.d_dim

        self.boundaries = boundaries
        self.real_boundaries = np.array(boundaries['real'])
        self.discrete_boundaries = np.array(boundaries['discrete'])

        self.objective_function_name = self.objective_function.__name__
        self.solution_history = None
        self.accuracy_history = None
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

        pos_d = np.zeros((population_size, self.d_dim)).astype(int)
        for col_index in range(self.d_dim):
            dd_lb, dd_ub = self.discrete_boundaries[col_index]
            while True:
                pos_d[:, col_index] = np.random.choice(a=range(dd_lb, dd_ub + 1), size=population_size)
                if sum(pos_d[:, col_index]) != 0:
                    break

        initial_pop = {'real': pos_r, 'discrete': pos_d}
        for sol in range(population_size):
            solution = {'real': pos_r[sol, :], 'discrete': pos_d[sol, :]}
            if not self.is_feasible(solution):
                initial_pop = self._get_initial_positions(population_size)
                break

        return initial_pop

    def optimize(self,
                 population_size: int,
                 iters: int,
                 r_power: int = 1,
                 elitist_check: bool = True,
                 chaotic_constant: bool = False,
                 repair_solution: bool = False,
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
            w_max (float): Maximum value of the chaotic term
            w_min (float): Minimum value of the chaotic term

        Returns:
            pd.DataFrame: Dataframe with the history of the optimization process
        """
        # Initializations
        vel_r = np.zeros((population_size, self.r_dim))
        vel_d = np.zeros((population_size, self.d_dim))
        vel = {'real': vel_r, 'discrete': vel_d}
        fit = np.zeros(population_size)
        mass = np.zeros(population_size)
        g_best = {'real': np.zeros(self.r_dim), 'discrete': np.zeros(self.d_dim)}
        g_best_score = float("-inf")
        best_acc = 0.0

        pos = self._get_initial_positions(population_size)

        best_solution_history = []
        best_accuracy_history = []
        convergence_curve = np.zeros(iters)

        print("GSA is optimizing  \"" + self.objective_function.__name__ + "\"")

        timer_start = time.time()
        self.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

        history = pd.DataFrame(columns=['Iteration', 'Fitness', 'Accuracy', 'ExecutionTime', 'Discrete', 'Real'])

        for current_iter in range(iters):
            for i in range(population_size):
                solution = {'real': pos['real'][i, :], 'discrete': pos['discrete'][i, :]}

                # Calculate objective function for each particle
                fitness, accuracy = self.objective_function(solution)
                fit[i] = fitness

                if fitness > g_best_score:
                    g_best_score = fitness
                    g_best = solution
                    best_acc = accuracy

            # Calculating Mass
            mass = mass_calculation(fit=fit)

            # Calculating Gravitational Constant
            gravity_constant = self._calculate_gravitational_constants(current_iter=current_iter,
                                                                       max_iters=iters,
                                                                       chaotic_constant=chaotic_constant,
                                                                       w_max=w_max,
                                                                       w_min=w_min)

            history.loc[len(history)] = [current_iter, g_best_score, best_acc, time.time() - timer_start, pos['discrete'], pos['real']]

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
            best_accuracy_history.append(best_acc)

            print(['At iteration ' + str(current_iter + 1) + ' the best fitness is ' + str(g_best_score)])

        timer_end = time.time()
        self.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.execution_time = timer_end - timer_start
        self.convergence = convergence_curve
        self.solution_history = best_solution_history
        self.accuracy_history = best_accuracy_history

        return history

    def _calculate_gravitational_constants(self,
                                           current_iter: int,
                                           max_iters: int,
                                           chaotic_constant: bool,
                                           w_max: float,
                                           w_min: float
                                           ) -> Mapping[str, float]:
        """
        Method to calculate the gravitational constants

        Args:
            current_iter (int): Current iteration
            max_iters (int): Maximum number of iterations
            chaotic_constant (bool): True if chaotic constant is used, False otherwise
            w_max (float): Maximum value of the chaotic term
            w_min (float): Minimum value of the chaotic term

        Returns:
            Mapping[str, float]: Gravitational constants for real and discrete variables
        """
        g_real = g_real_constant(current_iter, max_iters)
        g_discrete = g_bin_constant(current_iter, max_iters)
        # print("G_real: ", g_real)
        # print("G_discrete: ", g_discrete)

        if chaotic_constant:
            ch_value = w_max - current_iter * ((w_max - w_min) / max_iters)
            chaotic_term, _ = sin_chaotic_term(current_iter, ch_value)
            g_real += chaotic_term
            g_discrete += (chaotic_term / 100)
            # print("Chaotic Term: ", chaotic_term)

        return {'real': g_real, 'discrete': g_discrete}

    def _calculate_acceleration(self,
                                population_size: int,
                                pos: Mapping[str, np.ndarray],
                                mass: np.ndarray,
                                current_iter: int,
                                max_iters: int,
                                gravity_constant: Mapping[str, float],
                                r_power: int,
                                elitist_check: bool = True
                                ) -> Mapping[str, np.ndarray]:
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
            Mapping[str, np.ndarray]: Acceleration acting on the particles
        """
        acc_r = g_field(population_size=population_size,
                        dim=self.r_dim,
                        pos=pos['real'],
                        mass=mass,
                        current_iter=current_iter,
                        max_iters=max_iters,
                        gravity_constant=gravity_constant['real'],
                        r_power=r_power,
                        elitist_check=elitist_check,
                        real=True)

        acc_d = g_field(population_size=population_size,
                        dim=self.d_dim,
                        pos=pos['discrete'],
                        mass=mass,
                        current_iter=current_iter,
                        max_iters=max_iters,
                        gravity_constant=gravity_constant['discrete'],
                        r_power=r_power,
                        elitist_check=elitist_check,
                        real=False)

        return {'real': acc_r, 'discrete': acc_d}

    def _clip_positions(self,
                        pos: Mapping[str, np.ndarray],
                        individual: int
                        ) -> Mapping[str, np.ndarray]:
        """
        Clip the positions of the individuals to the boundaries of the search space

        Args:
            pos (Mapping[str, np.ndarray]): Current position of the particles
            individual (int): Index of the individual to be clipped

        Returns:
            Mapping[str, np.ndarray]: Clipped positions of the individuals
        """

        if self.r_dim > 0:
            l1_r = np.clip(pos['real'][individual, :], self.real_boundaries[:, 0], self.real_boundaries[:, 1])
        else:
            l1_r = np.array([])

        if self.d_dim > 0:
            l1_d = np.clip(pos['discrete'][individual, :], self.discrete_boundaries[:, 0],
                           self.discrete_boundaries[:, 1]).astype(int)
        else:
            l1_d = np.array([])

        pos['real'][individual, :] = l1_r
        pos['discrete'][individual, :] = l1_d

        return {'real': l1_r, 'discrete': l1_d}

    def _move(self,
              position: Mapping[str, np.ndarray],
              velocity: Mapping[str, np.ndarray],
              acceleration: Mapping[str, np.ndarray],
              population: int = 1,
              v_max: int = 6,
              repair_solution: bool = False
              ) -> Tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
        """
        Updates the position and velocity of particles in the search space based on their acceleration.
        This implementation leverages vectorized operations for efficiency.

        Args:
            position (Mapping[str, np.ndarray]): Current positions of the particles.
            velocity (Mapping[str, np.ndarray]): Current velocities of the particles.
            acceleration (Mapping[str, np.ndarray]): Current accelerations of the particles.
            population (int): Number of particles.
            v_max (int): Maximum velocity of the particles.
            repair_solution (bool): True if the solution should be repaired, False otherwise.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated position and velocity of the particles.
        """
        for i in range(population):
            # Update real variables
            r1 = np.random.random(position['real'][i].shape)  # Generate random coefficients for velocity update
            velocity['real'][i] = velocity['real'][i] * r1 + acceleration['real'][i]
            position['real'][i] = position['real'][i] + velocity['real'][i]  # Update position

            # Update discrete variables
            r2 = np.random.random(position['discrete'][i].shape)  # Generate random coefficients for velocity update
            velocity['discrete'][i] = velocity['discrete'][i] * r2 + acceleration['discrete'][i]
            velocity['discrete'][i] = np.clip(velocity['discrete'][i], a_min=None, a_max=v_max)

            discrete_move_probs = np.abs(np.tanh(velocity['discrete'][i]))  # Apply tanh activation function
            rand = np.random.rand(*discrete_move_probs.shape)

            position['discrete'][i][rand < discrete_move_probs] = 1 - position['discrete'][i][
                rand < discrete_move_probs]
            position['discrete'][i] = position['discrete'][i].astype(int)

            if not np.any(position['discrete'][i]):
                max_index = np.argmax(position['discrete'][i])
                position['discrete'][i, max_index] = 1

            new_solution = {'real': position['real'][i, :], 'discrete': position['discrete'][i, :]}

            if not self.is_feasible(new_solution):
                if not repair_solution:
                    new_solution = self._clip_positions(pos=new_solution, individual=i)
                else:
                    new_solution = self._repair_solution(solution=new_solution,
                                                         individual=i,
                                                         population=position)

            position['real'][i, :] = new_solution['real']
            position['discrete'][i, :] = new_solution['discrete']

        return position, velocity

    def _repair_solution(self,
                         solution: Mapping[str, np.ndarray],
                         individual: int,
                         population: Mapping[str, np.ndarray]
                         ) -> Mapping[str, np.ndarray]:
        """
        Repair the solution to make it feasible

        Args:
            solution (Mapping[str, np.ndarray]): Solution to be repaired.

        Returns:
            Mapping[str, np.ndarray]: Repaired solution.
        """
        print("#"*100)
        print("REPAIRING SOLUTION... ")
        print("#"*100)

        real_distances = {}
        discrete_distances = {}
        for i in range(len(population)):
            if i != individual:
                real_distances[i] = euclidean(solution['real'], population['real'][i])
                discrete_distances[i] = hamming(solution['discrete'], population['discrete'][i])

        sorted_real_distances = sorted(real_distances.items(), key=lambda x: x[1])
        sorted_discrete_distances = sorted(discrete_distances.items(), key=lambda x: x[1])

        # Unfeasible individual
        S_real = solution['real']
        S_discrete = solution['discrete']

        # Closest feasible individual
        R_real = population['real'][sorted_real_distances[0][0]]
        R_discrete = population['discrete'][sorted_discrete_distances[0][0]]

        S = {'real': S_real, 'discrete': S_discrete}
        while not self.is_feasible(S):
            X_real = []
            for i in range(len(S_real)):
                a = np.random.uniform(low=0, high=1)
                X_real.append(a * R_real[i] + (1 - a) * S_real[i])

            X_discrete = []
            for i in range(len(S_discrete)):
                a = np.random.uniform(low=0, high=1)
                X_discrete.append(round(float(a * R_discrete[i] + (1 - a) * S_discrete[i])))

            S = {'real': X_real, 'discrete': X_discrete}

        print("#" * 100)
        print("SOLUTION SUCCESSFULLY REPAIRED!!")
        print("#" * 100)
        return S
