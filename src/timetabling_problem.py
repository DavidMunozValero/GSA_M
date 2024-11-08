"""Infrastructure Manager Revenue maximization problem formulation."""

import datetime
import numpy as np

from copy import deepcopy
from functools import cache
from math import e, cos, pi
from pathlib import Path
from robin.services_generator.utils import build_service
from robin.supply.entities import TimeSlot, Line, Service, Supply
from typing import Any, List, Mapping, Tuple, Union

from ..benchmarks.utils import get_stations_positions
from .entities import Solution, Boundaries


class MPTT:
    """
    Class for the IM revenue maximization problem.
    """
    def __init__(self,
                 requested_schedule: Mapping[str, Mapping[str, Any]],
                 revenue_behaviour: Mapping[str, Mapping[str, float]],
                 line: Mapping[str, Tuple[float, float]],
                 safe_headway: int = 10,
                 max_stop_time: int = 10
                 ) -> None:
        """
        Constructor

        Args:
            requested_schedule (Mapping[str, List[str, Tuple[float, float]]]): requested schedule
            revenue_behaviour (Mapping[str, Mapping[str, float]]): revenue
            safe_headway (int): safe headway
            max_stop_time (int): max stop time
        """
        self.requested_schedule = requested_schedule
        self.line_stations = get_stations_positions(line)
        self.revenue = revenue_behaviour
        self.line = line

        self.safe_headway = safe_headway
        self.max_stop_time = max_stop_time

        self.n_services = len(self.requested_schedule)
        self.operational_times = self.get_operational_times()

        service_indexer = []
        reference_solution = []
        for service in requested_schedule:
            for sta_times in list(self.requested_schedule[service].values())[:-1]:
                reference_solution.append(sta_times[1])
                service_indexer.append(service)
        self.reference_solution = tuple(reference_solution)
        self.service_indexer = tuple(service_indexer)

        self.updated_schedule = deepcopy(self.requested_schedule)
        self.boundaries = self._calculate_boundaries()
        self.conflict_matrix = self._get_conflict_matrix()
        self.best_revenue = -np.inf
        self.best_solution = None
        self.feasible_schedules = []
        self.dt_indexer = self.get_departure_time_indexer()
        self.indexer = {sch: idx for idx, sch in enumerate(self.requested_schedule)}
        self.rev_indexer = {idx: sch for idx, sch in enumerate(self.requested_schedule)}
        self.requested_times = self.get_real_vars()
        self.scheduled_trains = np.zeros(self.n_services, dtype=np.bool_)

    def update_supply(self, path: Path,
                      solution: Solution
                      ) -> List[Service]:
        self.update_schedule(solution)
        services = []
        supply = Supply.from_yaml(path=path)
        scheduled_services = solution.discrete

        assert len(scheduled_services) == len(supply.services), "Scheduled services and services in supply do not match"

        for S_i, service in zip(scheduled_services, supply.services):
            if not S_i:
                continue

            service_schedule = self.updated_schedule[service.id]
            timetable = {sta: tuple(map(float, service_schedule[sta])) for sta in service_schedule}
            departure_time = list(timetable.values())[0][1]
            relative_timetable = {sta: tuple(map(lambda x: float(x - departure_time), service_schedule[sta])) for sta in service_schedule}
            updated_line_id = str(hash(str(relative_timetable.values())))
            updated_line = Line(updated_line_id, service.line.name, service.line.corridor, relative_timetable)
            date = service.date
            start_time = datetime.timedelta(minutes=float(departure_time))
            time_slot_id = f'{start_time.seconds}'

            updated_time_slot = TimeSlot(time_slot_id, start_time, start_time + datetime.timedelta(minutes=10))
            updated_service = build_service(id_=service.id,
                                            date=date,
                                            line=updated_line,
                                            time_slot=updated_time_slot,
                                            tsp=service.tsp,
                                            rs=service.rolling_stock,
                                            prices=service.prices,
                                            build_service_id=False)
            services.append(updated_service)
        return services

    def get_departure_time_indexer(self) -> Mapping[int, str]:
        """
        Build dictionary where keys are the index of the departure times and values are the service where the departure
        time belongs to.

        Returns:
            Mapping[int, str]: departure time indexer.
        """
        i = 0
        dt_indexer = {}
        for service in self.requested_schedule:
            for _ in range(len(self.requested_schedule[service]) - 1):
                dt_indexer[i] = service
                i += 1

        return dt_indexer

    def _calculate_boundaries(self) -> Boundaries:
        """
        Calculate boundaries for the departure times of the services.

        Returns:
            Boundaries: boundaries.
        """
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
                    max_dt_original = self.requested_schedule[service][stops[i]][1] + self.safe_headway
                    max_dt_updated = lower_bound + (self.max_stop_time - stop_time)
                    upper_bound = min(max_dt_original, max_dt_updated)
                boundaries.append([lower_bound, upper_bound])

        return Boundaries(real=boundaries, discrete=[])

    def _departure_time_feasibility(self, S_i: np.array) -> bool:
        """
        Check if there are any conflicts with the departure times.

        Args:
            S_i (np.array): solution

        Returns:
            bool: True if the departure time is feasible, False otherwise
        """
        S_i = np.array(S_i, dtype=np.bool_)
        if not np.any((S_i * self.conflict_matrix)[S_i]):
            return True
        return False

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

    def _get_conflict_matrix(self) -> np.array:
        """
        Get conflict matrix

        Returns:
            np.array: conflict matrix.
        """
        @cache
        def get_x_line_equation(A, B):
            x_coords = (A[0], B[0])
            y_coords = (A[1], B[1])
            m = (y_coords[1] - y_coords[0]) / (x_coords[1] - x_coords[0])
            c = y_coords[0] - m * x_coords[0]
            return lambda y: (y - c) / m

        def infer_times(service, station, origin: bool=True) -> float:
            idx = 1 if origin else 0
            if station in self.updated_schedule[service]:
                return self.updated_schedule[service][station][idx]

            stations = tuple(self.updated_schedule[service].keys())
            # Get station before and after
            stations_pos = [self.line_stations[s] for s in stations]
            # Get position of station
            station_pos = self.line_stations[station]
            # Get station before and after
            before = None
            after = None
            for i in range(len(stations)-1):
                if station_pos > stations_pos[i] and station_pos < stations_pos[i+1]:
                    before = stations[i]
                    after = stations[i+1]
                    break

            if not before or not after:
                raise ValueError(f"Station {station} not found in service {service}")
            A = (self.updated_schedule[service][before][1], self.line_stations[before])
            B = (self.updated_schedule[service][after][0], self.line_stations[after])
            line_other = get_x_line_equation(A, B)
            return line_other(self.line_stations[station])

        conflict_matrix = np.zeros((len(self.requested_schedule), len(self.requested_schedule)), dtype=np.bool_)

        for i, service in enumerate(self.requested_schedule):
            service_stations = tuple(self.requested_schedule[service].keys())
            for k, station in enumerate(service_stations):
                if k == len(service_stations) - 1:
                    break

                departure_station = station
                arrival_station = service_stations[k + 1]
                departure_time = self.updated_schedule[service][station][1]
                arrival_time = self.updated_schedule[service][service_stations[k + 1]][0]

                for j, other_service in enumerate(self.requested_schedule):
                    if other_service == service or conflict_matrix[i, j]:
                        continue

                    if tuple(self.requested_schedule[other_service].values())[0][1] > arrival_time:
                        continue
                    other_service_stations = tuple(self.requested_schedule[other_service].keys())

                    stations_between = []
                    for s in other_service_stations:
                        if self.line_stations[departure_station] <= self.line_stations[s] <= self.line_stations[arrival_station]:
                            stations_between.append(s)

                    if not stations_between:
                        continue
                    else:
                        # Get set of trips of other_service that could make a conflict with service
                        trips = set()
                        for s in stations_between:
                            idx = other_service_stations.index(s)
                            if 0 < idx < len(other_service_stations) - 1:
                                trips.add((other_service_stations[idx - 1], s))
                                trips.add((s, other_service_stations[idx + 1]))
                            elif idx == 0:
                                trips.add((s, other_service_stations[idx + 1]))
                            elif idx == len(other_service_stations) - 1:
                                trips.add((other_service_stations[idx - 1], s))

                    for trip in trips:
                        other_service_init, other_service_end = trip
                        inference_departure_station = max([departure_station, other_service_init],
                                                          key=lambda x: self.line_stations[x])
                        inference_arrival_station = min([arrival_station, other_service_end],
                                                        key=lambda x: self.line_stations[x])

                        other_departure_time = infer_times(other_service,
                                                           inference_departure_station,
                                                           origin=True)
                        other_arrival_time  = infer_times(other_service,
                                                          inference_arrival_station,
                                                          origin=False)

                        original_departure_time = infer_times(service, inference_departure_station, origin=True)
                        original_arrival_time = infer_times(service, inference_arrival_station, origin=False)

                        dt_gap = other_departure_time - original_departure_time
                        at_gap = other_arrival_time - original_arrival_time

                        same_sign = lambda x, y: x * y > 0
                        if same_sign(dt_gap, at_gap) and all(abs(t) >= 2* self.safe_headway for t in (dt_gap, at_gap)):
                            continue
                        else:
                            conflict_matrix[i, j] = True
                            conflict_matrix[j, i] = True

        return conflict_matrix

    def _travel_times_feasibility(self, S_i: np.array) -> bool:
        """
        Check if the travel times are feasible. In order to be feasible, the travel times must be greater than the
        requested ones.

        Returns:
            bool: True if the travel times are feasible, False otherwise
        """
        for i, service in enumerate(self.requested_schedule):
            if S_i[i] == 0:
                continue
            if not self.service_is_feasible(service):
                return False
        return True

    def get_best_schedule(self, solution: list) -> np.array:
        """
        Get the best feasible train scheduling based on revenue maximization.

        Args:
            solution (list): solution obtained from the optimization algorithm (timetable).

        Returns:
            np.array: best feasible train scheduling.
        """
        self.update_feasible_schedules(solution)
        best_schedule = None
        best_revenue = -np.inf
        for fs in self.feasible_schedules:
            revenue = self.get_revenue(Solution(real=solution, discrete=fs), update_schedule=False)
            if revenue > best_revenue:
                best_revenue = revenue
                best_schedule = fs
        return np.array(best_schedule)

    def get_heuristic_schedule(self) -> np.array:
        """
        Get best schedule using the heuristic approach.

        Returns:
            dict: best schedule

        CaSP: Conflict-avoiding Sequential Planner
        1) Schedule services without conflicts by checking the conflict matrices.
        2) Get dictionary of services with conflicts and their revenue based on the updated schedule
        3) While there are services with conflicts:
            3.1) Get service 's' with the best revenue, and schedule it
            3.2) Get set of services that have conflict with service 's'
            3.3) Update conflicts dictionary by removing services that have conflict with service 's'
                 (including 's')
        """
        # self.update_schedule(timetable)
        default_planner = np.array([(~cm).all() for cm in self.conflict_matrix], dtype=np.bool_)
        conflicts = set(sch for sch in self.updated_schedule if not default_planner[self.indexer[sch]])
        conflicts_revenue = {sc: self.get_service_revenue(sc) for sc in conflicts}
        conflicts_revenue = dict(sorted(conflicts_revenue.items(), key=lambda item: item[1]))

        while conflicts_revenue:
            # Get service 's' with the best revenue, and schedule it
            s, _ = conflicts_revenue.popitem()
            default_planner[self.indexer[s]] = True

            # Get set of services that have conflict with service 's'
            conflicts_with_s = np.where(self.conflict_matrix[self.indexer[s]])[0]
            conflicts_with_s = set(self.rev_indexer[conflict] for conflict in conflicts_with_s)

            # Update conflicts dictionary by removing services that have conflict with service 's' (including 's')
            conflicts_revenue = dict(filter(lambda p: p[0] not in conflicts_with_s, conflicts_revenue.items()))

        return default_planner

    def objective_function(self,
                           solution: List[float]) -> float:
        """
        Get fitness

        Args:
            timetable (Solution): solution
            heuristic_schedule (bool): heuristic schedule

        Returns:
            Tuple[float, int]: fitness and accuracy (0)
        """
        solution = np.array(solution, dtype=np.int32)
        self.update_schedule(solution)
        schedule = self.get_heuristic_schedule()
        return self.get_revenue(Solution(real=solution, discrete=schedule))

    def _update_dynamic_bounds(self, j, ot_idx, proposed_times, updated_boundaries):
        if j != len(self.requested_times) - 1 and self.dt_indexer[j + 1] == self.dt_indexer[j]:
            travel_time = self.operational_times[self.dt_indexer[j]][ot_idx]
            stop_time = self.operational_times[self.dt_indexer[j]][ot_idx + 1]
            ot_idx += 2
            lower_bound = proposed_times[j] + travel_time + stop_time
            max_dt_original = self.requested_times[j + 1] + self.safe_headway
            max_dt_updated = lower_bound + (self.max_stop_time - stop_time)
            upper_bound = min(max_dt_original, max_dt_updated)
            updated_boundaries[j + 1] = (lower_bound, upper_bound)
        else:
            ot_idx = 0

        return ot_idx, updated_boundaries

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
            for i in range(len(stops) - 1):
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

    def get_real_vars(self) -> List[int]:
        """
        Get real variables

        Returns:
            Tuple[List[int], List[List[int]]]: real variables and boundaries
        """
        real_vars = []

        for service in self.requested_schedule:
            stops = list(self.requested_schedule[service].keys())
            for i in range(len(stops) - 1):
                real_vars.append(self.requested_schedule[service][stops[i]][1])

        return real_vars

    def get_service_revenue(self, service):
        k = self.revenue[service]['k']
        departure_station = list(self.requested_schedule[service].keys())[0]
        departure_time_delta = abs(self.updated_schedule[service][departure_station][1] -
                                   self.requested_schedule[service][departure_station][1])
        tt_penalties = []
        for j, stop in enumerate(self.requested_schedule[service].keys()):
            if j == 0 or j == len(self.requested_schedule[service]) - 1:
                continue
            tt_penalty = self.penalty_function(abs(
                self.updated_schedule[service][stop][1] - self.requested_schedule[service][stop][
                    1]) / self.safe_headway, k)
            tt_penalties.append(tt_penalty * self.revenue[service]['tt_max_penalty'])
        dt_penalty = self.penalty_function(departure_time_delta / self.safe_headway, k) * self.revenue[service]['dt_max_penalty']
        return self.revenue[service]['canon'] - dt_penalty - np.sum(tt_penalties)

    def service_is_feasible(self, service: str) -> bool:
        original_service_times = tuple(self.requested_schedule[service].values())
        updated_service_times = tuple(self.updated_schedule[service].values())
        for j in range(len(original_service_times) - 1):
            # Travel time feasibility
            original_tt = original_service_times[j + 1][0] - original_service_times[j][1]
            updated_tt = updated_service_times[j + 1][0] - updated_service_times[j][1]
            if updated_tt < original_tt:
                return False
            if j == 0:
                continue
            # Stop time feasibility
            original_st = original_service_times[j][1] - original_service_times[j][0]
            updated_st = updated_service_times[j][1] - updated_service_times[j][0]
            if updated_st < original_st:
                return False
        return True


    def get_revenue(self,
                    solution: Solution
                    ) -> float:
        """
        Get IM revenue.

        Args:
            solution (Solution): solution

        Returns:
            float: IM revenue
        """
        S_i = solution.discrete

        im_revenue = 0.0
        for i, service in enumerate(self.requested_schedule):
            if S_i[i] and self.service_is_feasible(service):
                im_revenue += self.get_service_revenue(service)

        if im_revenue > self.best_revenue:
            self.best_revenue = im_revenue
            self.best_solution = solution

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

        if dt_feasible and tt_feasible:
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

    @cache
    def truth_table(self, dim: int):
        if dim < 1:
            return [[]]
        sub_tt = self.truth_table(dim - 1)
        return [row + [val] for row in sub_tt for val in [0, 1]]

    def update_feasible_schedules(self, solution: List[float]):
        """
        Get feasible scheduling

        Args:
            timetable (Solution): timetable
        """
        self.update_schedule(solution)
        train_combinations = self.truth_table(dim=self.n_services)
        self.feasible_schedules = list(filter(lambda S_i: self._departure_time_feasibility(S_i), train_combinations))

    def update_schedule(self, solution: np.array):
        """
        Update schedule with the solution

        Args:
            solution (Solution): solution
        """
        departure_times = solution if solution.any() else self.get_real_vars()
        dt_idx = 0
        for i, service in enumerate(self.updated_schedule):
            ot_idx = 0
            for j, stop in enumerate(self.updated_schedule[service]):
                if j == 0:
                    departure_time = departure_times[dt_idx]
                    arrival_time = departure_time
                    dt_idx += 1
                elif j == len(self.updated_schedule[service]) - 1:
                    arrival_time = departure_times[dt_idx - 1] + self.operational_times[service][ot_idx]
                    departure_time = arrival_time
                else:
                    arrival_time = departure_times[dt_idx - 1] + self.operational_times[service][ot_idx]
                    departure_time = departure_times[dt_idx]
                    ot_idx += 2
                    dt_idx += 1

                self.updated_schedule[service][stop][0] = arrival_time
                self.updated_schedule[service][stop][1] = departure_time

        self.boundaries = self._calculate_boundaries()
        self.conflict_matrix = self._get_conflict_matrix()
