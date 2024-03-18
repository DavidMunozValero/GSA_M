import numpy as np

from copy import deepcopy
from functools import cache
from math import e, cos, pi
from typing import Mapping, Tuple, Union, List

from src.entities import Solution, Boundaries


class RevenueMaximization:
    """
    Class for the revenue maximization problem.
    """

    def __init__(self,
                 requested_schedule: Mapping[int, Mapping[str, Union[int, int]]],
                 revenue: Mapping[int, Mapping[str, Union[float, int]]],
                 safe_headway: int = 10,
                 max_stop_time: int = 10
                 ) -> None:
        """
        Constructor

        Args:
            requested_schedule (dict): requested schedule
        """
        self.requested_schedule = requested_schedule
        self.revenue = revenue
        self.safe_headway = safe_headway
        self.max_stop_time = max_stop_time

        self.operational_times = self.get_operational_times()
        self._updated_schedule = deepcopy(requested_schedule)
        self.requested_times = self.get_real_vars()
        self._boundaries = self._calculate_boundaries()
        self.feasible_schedules = []
        self.scheduled_trains = np.zeros(len(self.requested_schedule))

    def _calculate_boundaries(self):
        boundaries = []

        for service in self.requested_schedule:
            stops = list(self.requested_schedule[service].keys())
            ot_idx = 0
            for i in range(len(stops) - 1):
                if i == 0:
                    lower_bound = self.requested_schedule[service][stops[i]][1] - self.safe_headway
                    upper_bound = self.requested_schedule[service][stops[i]][1] + self.safe_headway
                else:
                    travel_time = self.operational_times[service][ot_idx]
                    stop_time = self.operational_times[service][ot_idx + 1]
                    ot_idx += 2
                    lower_bound = self.updated_schedule[service][stops[i - 1]][1] + travel_time + stop_time
                    upper_bound = self.requested_schedule[service][stops[i]][1] + self.safe_headway
                boundaries.append([lower_bound, upper_bound])

        return Boundaries(real=boundaries, discrete=[])

    @property
    def updated_schedule(self):
        return self._updated_schedule

    @updated_schedule.setter
    def updated_schedule(self, value):
        self._updated_schedule = value
        self._boundaries = self._calculate_boundaries()

    @property
    def boundaries(self):
        return self._boundaries

    def _departure_time_feasibility(self, scheduling) -> bool:
        """
        Check if there are any conflicts with the departure times.

        Args:
            solution (dict): solution

        Returns:
            bool: True if the departure time is feasible, False otherwise
        """
        S_i = scheduling

        # Get conflicts between services
        for i, service in enumerate(self.updated_schedule):
            if S_i[i] == 0:
                continue
            for j, service_k in enumerate(self.updated_schedule):
                if i == j or S_i[j] == 0:
                    continue
                for stop in self.updated_schedule[service]:
                    if service_k == service or stop not in self.updated_schedule[service_k]:
                        continue

                    if abs(self.updated_schedule[service][stop][1] - self.updated_schedule[service_k][stop][
                        1]) < self.safe_headway:
                        if S_i[i] and S_i[j]:
                            return False
        return True

    def _feasible_boundaries(self, solution: Solution) -> bool:
        """
        Check if the solution is within the boundaries

        Args:
            solution (dict): solution

        Returns:
            bool: True if the solution is within the boundaries, False otherwise
        """
        for i, rv in enumerate(solution.real):
            if rv < self.boundaries.real[i][0] or rv > self.boundaries.real[i][1]:
                return False
        return True

    def _stop_times_feasibility(self, scheduling):
        """
        Check if the travel times are feasible. In order to be feasible, the travel times must be greater than the requested ones.

        Args:
            solution (dict): solution

        Returns:
            bool: True if the travel times are feasible, False otherwise
        """
        S_i = scheduling
        for i, service in enumerate(self.requested_schedule):
            if S_i[i] == 0:
                continue
            original_service_times = tuple(self.requested_schedule[service].values())
            updated_service_times = tuple(self.updated_schedule[service].values())
            for j in range(1, len(original_service_times) - 1):
                original_st = original_service_times[j][1] - original_service_times[j][0]
                updated_st = updated_service_times[j][1] - updated_service_times[j][0]
                if updated_st < original_st:
                    return False
        return True

    def _travel_times_feasibility(self, scheduling) -> bool:
        """
        Check if the travel times are feasible. In order to be feasible, the travel times must be greater than the requested ones.

        Returns:
            bool: True if the travel times are feasible, False otherwise
        """
        S_i = scheduling
        for i, service in enumerate(self.requested_schedule):
            if S_i[i] == 0:
                continue
            original_service_times = tuple(self.requested_schedule[service].values())
            updated_service_times = tuple(self.updated_schedule[service].values())
            for j in range(len(original_service_times) - 1):
                original_tt = original_service_times[j + 1][0] - original_service_times[j][1]
                updated_tt = updated_service_times[j + 1][0] - updated_service_times[j][1]
                if updated_tt < original_tt:
                    return False
        return True

    def feasible_services_times(self, timetable):
        """
        Check if the service is feasible

        Returns:
            bool: True if the service is feasible, False otherwise
        """
        self.update_schedule(timetable)
        if not self._feasible_boundaries(timetable):
            return False

        scheduling = np.ones(len(self.requested_schedule))
        if self._travel_times_feasibility(scheduling) and self._stop_times_feasibility(scheduling):
            return True
        return False

    def get_best_schedule(self, timetable: Solution):
        """
        Get best schedule

        Args:
            timetable (Solution): solution

        Returns:
            dict: best schedule
        """
        self.update_feasible_schedules(timetable)
        best_schedule = None
        best_revenue = -np.inf
        for fs in self.feasible_schedules:
            revenue = self.get_revenue(Solution(real=timetable.real, discrete=fs), update_schedule=False)
            if revenue > best_revenue:
                best_revenue = revenue
                best_schedule = fs
        return np.array(best_schedule)

    def get_heuristic_schedule(self, timetable: Solution):
        """
        Get best schedule

        Args:
            timetable (Solution): solution

        Returns:
            dict: best schedule
        """
        self.update_feasible_schedules(timetable)
        scheduled_trains = sorted(self.feasible_schedules, key=lambda x: np.sum(x), reverse=True)
        return scheduled_trains[0]

    def get_fitness_gsa(self,
                        timetable: Solution,
                        heuristic_schedule: bool = True):
        """
        Get fitness

        Args:
            timetable (Solution): solution
            heuristic_schedule (bool): heuristic schedule

        Returns:
            Tuple[float, int]: fitness and number of evaluations
        """
        if not heuristic_schedule:
            schedule = self.get_best_schedule(timetable)
        else:
            schedule = self.get_heuristic_schedule(timetable)
        return self.get_revenue(Solution(real=timetable.real, discrete=schedule), update_schedule=False), 0

    def get_initial_population(self, population_size: int) -> List[Solution]:
        """
        Get initial population

        Returns:
            dict: initial population
        """
        identify_services = [service for service in self.requested_schedule for _ in
                             range(len(self.requested_schedule[service]) * 2 - 2)]
        population = []
        for _ in range(population_size):
            proposed_times = deepcopy(self.requested_times)
            updated_boundaries = deepcopy(self.boundaries.real)
            for j in range(len(self.requested_times)):
                lower_bound, upper_bound = updated_boundaries[j]
                proposed_times[j] = np.round(np.random.uniform(lower_bound, upper_bound), 2)
                if j != len(self.requested_times) - 1 and identify_services[j + 1] == identify_services[j]:
                    travel_time = self.requested_times[j + 1] - self.requested_times[j]
                    updated_boundaries[j + 1] = (proposed_times[j] + travel_time, updated_boundaries[j + 1][1])

            population.append(Solution(real=np.array(proposed_times, dtype=float), discrete=np.array([])))

        return population

    def get_operational_times(self):
        """
        Get operational times

        Returns:
            list: operational times
        """
        operational_times = {}
        for service in self.requested_schedule:
            stops = list(self.requested_schedule[service].keys())
            operational_times[service] = []
            for i in range(len(stops)-1):
                origin = stops[i]
                destination = stops[i + 1]
                arrival_time = self.requested_schedule[service][destination][0]
                departure_time = self.requested_schedule[service][origin][1]
                travel_time = arrival_time - departure_time

                if i == 0:
                    operational_times[service].append(travel_time)
                else:
                    stop_arrival_time = self.requested_schedule[service][origin][0]
                    stop_time = departure_time - stop_arrival_time
                    operational_times[service] += [stop_time, travel_time]

        return operational_times

    def get_real_vars(self) -> List[float]:
        """
        Get real variables

        Returns:
            Tuple[List[float], List[List[float]]]: real variables and boundaries
        """
        real_vars = []

        for service in self.requested_schedule:
            stops = list(self.requested_schedule[service].keys())
            for i in range(len(stops)-1):
                real_vars.append(self.requested_schedule[service][stops[i]][1])

        return real_vars

    def get_revenue(self,
                    solution: Solution,
                    update_schedule: bool = True) -> int:
        """
        Get IM revenue.

        Args:
            solution (Solution): solution
            update_schedule (bool): update schedule

        Returns:
            float: IM revenue
        """
        if update_schedule:
            self.update_schedule(solution)
        S_i = solution.discrete

        im_revenue = 0
        for i, service in enumerate(self.requested_schedule):
            k = self.revenue[service]['k']
            departure_station = list(self.requested_schedule[service].keys())[0]
            departure_time_delta = abs(self.updated_schedule[service][departure_station][1] -
                                       self.requested_schedule[service][departure_station][1])
            tt_penalties = []
            for j, stop in enumerate(self.requested_schedule[service].keys()):
                if i == 0 or i == len(self.requested_schedule[service]) - 1:
                    continue
                tt_penalty = self.penalty_function(abs(
                    self.updated_schedule[service][stop][1] - self.requested_schedule[service][stop][
                        1]) / self.safe_headway, k)
                tt_penalties.append(tt_penalty * self.revenue[service]['tt_max_penalty'])
            dt_penalty = self.penalty_function(departure_time_delta / self.safe_headway, k) * self.revenue[service][
                'dt_max_penalty']
            im_revenue += self.revenue[service]['canon'] * S_i[i] - dt_penalty * S_i[i] - np.sum(tt_penalties) * S_i[i]

        return im_revenue

    def is_feasible(self,
                    timetable: Solution,
                    scheduling: np.array,
                    update_schedule: bool = True
                    ) -> bool:
        """
        Check if the solution is feasible

        Args:
            timetable (Solution): solution obtained from the optimization algorithm
            scheduling (np.array): scheduling
            update_schedule (bool): update schedule

        Returns:
            bool: True if the solution is feasible, False otherwise
        """
        if update_schedule:
            self.update_schedule(timetable)

        if not self._feasible_boundaries(timetable):
            return False

        dt_feasible = self._departure_time_feasibility(scheduling)
        tt_feasible = self._travel_times_feasibility(scheduling)
        st_feasible = self._stop_times_feasibility(scheduling)
        if all([dt_feasible, tt_feasible, st_feasible]):
            return True
        return False

    @staticmethod
    def penalty_function(x: float, k: int) -> float:
        """
        Penalty function

        Args:
            x (float): x
            k (int): k

        Returns:
            float: penalty
        """
        return 1 - e ** (-k * x ** 2) * ((1 / 2) * cos(pi * x) + (1 / 2))

    def update_feasible_schedules(self, timetable: Solution):
        """
        Get feasible scheduling

        Args:
            timetable (Solution): timetable
        """
        @cache
        def truth_table(dim: int):
            if dim < 1:
                return [[]]
            sub_tt = truth_table(dim - 1)
            return [row + [val] for row in sub_tt for val in [0, 1]]

        self.update_schedule(timetable)

        train_combinations = truth_table(len(self.requested_schedule))
        self.feasible_schedules = list(
            filter(lambda S_i: self._departure_time_feasibility(S_i), train_combinations))

    def update_schedule(self, solution: Solution):
        """
        Update schedule with the solution

        Args:
            solution (Solution): solution
        """
        departure_times = solution.real if solution.real.size else self.get_real_vars()
        dt_idx = 0
        for i, service in enumerate(self.updated_schedule):
            ot_idx = 0
            for j, stop in enumerate(self.updated_schedule[service]):
                if j == 0:
                    departure_time = departure_times[dt_idx]
                    arrival_time = departure_time
                    dt_idx += 1
                elif j == len(self.updated_schedule[service]) - 1:
                    arrival_time = departure_times[dt_idx-1] + self.operational_times[service][ot_idx]
                    departure_time = arrival_time
                    ot_idx += 2
                else:
                    arrival_time = departure_times[dt_idx-1] + self.operational_times[service][ot_idx]
                    departure_time = departure_times[dt_idx]
                    ot_idx += 2
                    dt_idx += 1

                self.updated_schedule[service][stop][0] = arrival_time
                self.updated_schedule[service][stop][1] = departure_time

        self._boundaries = self._calculate_boundaries()
