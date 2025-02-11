import datetime
import numpy as np
from copy import deepcopy
from functools import cache
from math import e, cos, pi
from pathlib import Path
from typing import Any, List, Mapping, Tuple, Union

from robin.services_generator.utils import build_service
from robin.supply.entities import TimeSlot, Line, Service, Supply

from benchmarks.utils import get_stations_positions
from .entities import Solution, Boundaries


class MPTT:
    """
    Infrastructure Manager Revenue Maximization Problem Formulation.

    This class formulates and solves the revenue maximization problem for train scheduling.
    It maintains both the requested and updated schedules, computes operational times,
    enforces feasibility (via boundaries and conflict matrices) and evaluates the revenue,
    including equity considerations.
    """

    def __init__(
            self,
            requested_schedule: Mapping[str, Mapping[str, Any]],
            revenue_behavior: Mapping[str, Mapping[str, float]],
            line: Mapping[str, Tuple[float, float]],
            safe_headway: int = 10,
            max_stop_time: int = 10,
            fair_index: Union[None, str] = None,
    ) -> None:
        """
        Initialize the MPTT instance.

        Args:
            requested_schedule: The requested schedule mapping.
            revenue_behavior: The revenue behavior parameters.
            line: Mapping of line station positions.
            safe_headway: The minimum safe headway time between trains.
            max_stop_time: Maximum allowed stop time.
            fair_index: The fairness index to use for equity considerations.
        """
        self.requested_schedule = requested_schedule
        self.line_stations = get_stations_positions(line)
        self.revenue = revenue_behavior
        self.line = line

        self.safe_headway = safe_headway
        self.im_mod_margin = 60
        self.max_stop_time = max_stop_time
        self.fair_index = fair_index

        if self.fair_index == "Jain":
            self.fairness_index = self.jains_fairness_index
        elif self.fair_index == "Gini":
            self.fairness_index = self.gini_fairness_index
        elif self.fair_index == "Atkinson":
            self.fairness_index = self.atkinson_fairness_index
        else:
            self.fairness_index = None

        self.n_services = len(self.requested_schedule)
        self.operational_times = self.get_operational_times()
        self.services_by_ru = self.get_n_services_by_ru()
        self.capacities = self.get_capacities()

        # Build reference solution and service indexer (for real variables)
        reference_solution = []
        service_indexer = []
        for service in self.requested_schedule:
            stops_times = list(self.requested_schedule[service].values())[:-1]
            for sta_times in stops_times:
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
        self.scheduled_trains = np.zeros(self.n_services, dtype=bool)

    # === Public Interface Methods ===

    def update_supply(self, path: Path, solution: Solution) -> List[Service]:
        """
        Update the supply based on the provided solution.

        Args:
            path: Path to the YAML file containing the supply.
            solution: The solution containing discrete scheduling decisions.

        Returns:
            A list of updated Service objects.
        """
        self.update_schedule(solution)
        services = []
        supply = Supply.from_yaml(path=path)
        scheduled_services = solution.discrete

        if len(scheduled_services) != len(supply.services):
            raise AssertionError("Scheduled services and services in supply do not match")

        for S_i, service in zip(scheduled_services, supply.services):
            if not S_i:
                continue

            service_schedule = self.updated_schedule[service.id]
            # Convert schedule times to floats
            timetable = {sta: tuple(map(float, times)) for sta, times in service_schedule.items()}
            departure_time = list(timetable.values())[0][1]
            # Calculate timetable relative to the departure time
            relative_timetable = {
                sta: tuple(float(t) - departure_time for t in times)
                for sta, times in service_schedule.items()
            }
            updated_line_id = str(hash(str(list(relative_timetable.values()))))
            updated_line = Line(updated_line_id, service.line.name, service.line.corridor, relative_timetable)
            date = service.date
            start_time = datetime.timedelta(minutes=float(departure_time))
            time_slot_id = f"{start_time.seconds}"
            updated_time_slot = TimeSlot(time_slot_id, start_time, start_time + datetime.timedelta(minutes=10))
            updated_service = build_service(
                id_=service.id,
                date=date,
                line=updated_line,
                time_slot=updated_time_slot,
                tsp=service.tsp,
                rs=service.rolling_stock,
                prices=service.prices,
                build_service_id=False,
            )
            services.append(updated_service)
        return services

    def update_schedule(self, solution: np.array) -> None:
        """
        Update the schedule using the provided solution.

        Args:
            solution: Array of departure times (real variables) for scheduling.
        """
        departure_times = solution if solution.any() else self.get_real_vars()
        dt_idx = 0
        for service in self.updated_schedule:
            ot_idx = 0
            stops = list(self.updated_schedule[service].keys())
            for j, stop in enumerate(stops):
                if j == 0:
                    departure_time = departure_times[dt_idx]
                    arrival_time = departure_time
                    dt_idx += 1
                elif j == len(stops) - 1:
                    arrival_time = departure_times[dt_idx - 1] + self.operational_times[service][ot_idx]
                    departure_time = arrival_time
                else:
                    arrival_time = departure_times[dt_idx - 1] + self.operational_times[service][ot_idx]
                    departure_time = departure_times[dt_idx]
                    ot_idx += 2
                    dt_idx += 1

                self.updated_schedule[service][stop][0] = arrival_time
                self.updated_schedule[service][stop][1] = departure_time

        # Recalculate boundaries and conflict matrix after updating times
        self.boundaries = self._calculate_boundaries()
        self.conflict_matrix = self._get_conflict_matrix()

    def update_feasible_schedules(self, solution: List[float]) -> None:
        """
        Update feasible schedules based on the provided solution.

        Args:
            solution: List of departure times (real variables).
        """
        self.update_schedule(solution)
        # Generate all possible binary schedules (truth table) for n_services
        train_combinations = self.truth_table(dim=self.n_services)
        self.feasible_schedules = [S_i for S_i in train_combinations if self._departure_time_feasibility(S_i)]

    def objective_function(self, solution: List[float]) -> float:
        """
        Compute the fitness (objective value) for the provided solution.
        If 'equity' is True, the revenue is multiplied by Jain's fairness index.

        Args:
            solution: List of departure times.

        Returns:
            Fitness value (float).
        """
        solution_arr = np.array(solution, dtype=np.int32)
        self.update_schedule(solution_arr)
        if self.fairness_index:
            schedule = self.get_heuristic_schedule_new()
            fairness, _ = self.fairness_index(schedule, self.capacities)
        else:
            schedule = self.get_heuristic_schedule_old()
            fairness = 1.0
        return self.get_revenue(Solution(real=solution, discrete=schedule)) * fairness

    def get_revenue(self, solution: Solution) -> float:
        """
        Compute the total revenue for the given solution.

        Args:
            solution: A Solution object containing real and discrete scheduling decisions.

        Returns:
            Total revenue (float).
        """
        S_i = solution.discrete
        im_revenue = 0.0
        for idx, service in enumerate(self.requested_schedule):
            if S_i[idx] and self.service_is_feasible(service):
                im_revenue += self.get_service_revenue(service)

        if im_revenue > self.best_revenue:
            self.best_revenue = im_revenue
            self.best_solution = solution

        return im_revenue

    def is_feasible(
            self, timetable: Solution, scheduling: np.array, update_schedule: bool = True
    ) -> bool:
        """
        Check if the provided solution is feasible.

        Args:
            timetable: The solution obtained from the optimization algorithm.
            scheduling: Boolean array representing discrete scheduling decisions.
            update_schedule: Whether to update the schedule with the provided timetable.

        Returns:
            True if the solution is feasible, False otherwise.
        """
        if update_schedule:
            self.update_schedule(timetable)

        if not self._feasible_boundaries(timetable):
            return False

        dt_feasible = self._departure_time_feasibility(scheduling)
        tt_feasible = self._travel_times_feasibility(scheduling)
        return dt_feasible and tt_feasible

    def get_best_schedule(self, solution: List[float]) -> np.array:
        """
        Determine the best feasible schedule based on revenue maximization.

        Args:
            solution: List of departure times from the optimization algorithm.

        Returns:
            Best feasible schedule as a numpy array.
        """
        self.update_feasible_schedules(solution)
        best_schedule = None
        best_revenue = -np.inf
        for fs in self.feasible_schedules:
            revenue = self.get_revenue(Solution(real=solution, discrete=fs))
            if revenue > best_revenue:
                best_revenue = revenue
                best_schedule = fs
        return np.array(best_schedule) if best_schedule is not None else np.array([])

    def get_heuristic_schedule_new(self) -> np.array:
        """
        Compute the best schedule using a new heuristic based on improving fairness
        for the most disadvantaged RU.

        Returns:
            Final schedule as a boolean numpy array.
        """
        # Initially, schedule services that do not have any conflicts.
        default_planner = np.array([not cm.any() for cm in self.conflict_matrix], dtype=bool)
        # master_conflicts: set of services not yet scheduled.
        master_conflicts = {sch for sch in self.updated_schedule if not default_planner[self.indexer[sch]]}

        while master_conflicts:
            fair_index, ratios = self.fairness_index(default_planner, self.capacities)
            # Find, among pending services, those belonging to the RU with the worst ratio.
            conflicts = set()
            for ru in sorted(ratios, key=ratios.get):
                conflicts = {sch for sch in master_conflicts if self.revenue[sch]["ru"] == ru}
                if conflicts:
                    break

            if not conflicts:
                break

            # Evaluate the fairness improvement if a candidate is scheduled.
            conflicts_equity = {
                sc: self.get_service_equity(default_planner.copy(), sc) for sc in conflicts
            }
            # Select the service with the greatest improvement.
            s = max(conflicts_equity, key=conflicts_equity.get)
            default_planner[self.indexer[s]] = True
            master_conflicts.discard(s)
            # Remove services that conflict with the newly scheduled service.
            conflicts_with_s = {self.rev_indexer[idx] for idx in np.where(self.conflict_matrix[self.indexer[s]])[0]}
            master_conflicts -= conflicts_with_s

        return default_planner

    def get_heuristic_schedule_old(self) -> np.array:
        """
        Compute the best schedule using an older (conflict‐avoiding sequential) heuristic.

        Returns:
            Final schedule as a boolean numpy array.
        """
        default_planner = np.array([not cm.any() for cm in self.conflict_matrix], dtype=bool)
        conflicts = {sch for sch in self.updated_schedule if not default_planner[self.indexer[sch]]}
        conflicts_revenue = {sc: self.get_service_revenue(sc) for sc in conflicts}
        # Sort by revenue (lowest first)
        conflicts_revenue = dict(sorted(conflicts_revenue.items(), key=lambda item: item[1]))

        while conflicts_revenue:
            # Select the service with the highest revenue among the conflicts.
            s = next(reversed(conflicts_revenue))
            default_planner[self.indexer[s]] = True
            # Eliminar s de conflicts_revenue para evitar bucle infinito.
            conflicts_revenue.pop(s, None)
            conflicts_with_s = {self.rev_indexer[idx] for idx in np.where(self.conflict_matrix[self.indexer[s]])[0]}
            conflicts_revenue = {k: v for k, v in conflicts_revenue.items() if k not in conflicts_with_s}
        return default_planner

    def get_service_equity(self, scheduled: np.array, service: str) -> float:
        """
        Compute the fairness index if the given service were scheduled.

        Args:
            scheduled: Current boolean scheduling array.
            service: Service identifier.

        Returns:
            Fairness index after scheduling the service.
        """
        scheduled[list(self.updated_schedule.keys()).index(service)] = True
        fair_index, _ = self.fairness_index(scheduled, self.capacities)
        return fair_index

    def get_operational_times(self) -> Mapping[str, List[float]]:
        """
        Compute operational times for each service based on the requested schedule.

        Returns:
            A mapping from service to a list of operational times.
        """
        operational_times = {}
        for service, stops in self.requested_schedule.items():
            stop_keys = list(stops.keys())
            times = []
            for i in range(len(stop_keys) - 1):
                origin = stop_keys[i]
                destination = stop_keys[i + 1]
                travel_time = stops[destination][0] - stops[origin][1]
                if i == 0:
                    times.append(travel_time)
                else:
                    stop_time = stops[origin][1] - stops[origin][0]
                    times.extend([stop_time, travel_time])
            operational_times[service] = times
        return operational_times

    def get_real_vars(self) -> List[int]:
        """
        Extract the real variables (departure times) from the requested schedule.

        Returns:
            List of departure times.
        """
        real_vars = []
        for service, stops in self.requested_schedule.items():
            stop_keys = list(stops.keys())
            for i in range(len(stop_keys) - 1):
                real_vars.append(stops[stop_keys[i]][1])
        return real_vars

    def get_service_revenue(self, service: str) -> float:
        """
        Compute the revenue for a given service based on its updated schedule.

        Args:
            service: Service identifier.

        Returns:
            Revenue value (float) for the service.
        """
        k = self.revenue[service]["k"]
        departure_station = list(self.requested_schedule[service].keys())[0]
        departure_time_delta = abs(
            self.updated_schedule[service][departure_station][1] -
            self.requested_schedule[service][departure_station][1]
        )
        tt_penalties = []
        stop_keys = list(self.requested_schedule[service].keys())
        for j, stop in enumerate(stop_keys):
            if j == 0 or j == len(stop_keys) - 1:
                continue
            penalty_val = self.penalty_function(
                abs(self.updated_schedule[service][stop][1] - self.requested_schedule[service][stop][
                    1]) / self.im_mod_margin,
                k,
            )
            tt_penalties.append(penalty_val * self.revenue[service]["tt_max_penalty"])
        dt_penalty = self.penalty_function(departure_time_delta / self.im_mod_margin, k) * self.revenue[service][
            "dt_max_penalty"]
        return self.revenue[service]["canon"] - dt_penalty - np.sum(tt_penalties)

    def service_is_feasible(self, service: str) -> bool:
        """
        Check if the updated schedule for a service is feasible relative to its requested schedule.

        Args:
            service: Service identifier.

        Returns:
            True if the service schedule is feasible, False otherwise.
        """
        original_times = list(self.requested_schedule[service].values())
        updated_times = list(self.updated_schedule[service].values())
        for j in range(len(original_times) - 1):
            original_tt = original_times[j + 1][0] - original_times[j][1]
            updated_tt = updated_times[j + 1][0] - updated_times[j][1]
            if updated_tt < original_tt:
                return False
            if j > 0:
                original_st = original_times[j][1] - original_times[j][0]
                updated_st = updated_times[j][1] - updated_times[j][0]
                if updated_st < original_st:
                    return False
        return True

    def jains_fairness_index(
            self, bool_scheduled: List[bool], capacities: Mapping[Any, float]
    ) -> Tuple[float, Mapping[Any, float]]:
        """
        Calculate the weighted Jain's fairness index based on the scheduled resources and capacities.

        Args:
            bool_scheduled: Boolean list indicating which services are scheduled.
            capacities: Mapping of capacity values for each RU.

        Returns:
            A tuple containing:
              - The Jain's fairness index (float).
              - A mapping of resource-to-capacity ratios.
        """
        scheduled = {}
        for service, scheduled_flag in zip(self.revenue.keys(), bool_scheduled):
            ru = self.revenue[service]["ru"]
            scheduled[ru] = scheduled.get(ru, 0) + (self.revenue[service]["importance"] if scheduled_flag else 0)

        for ru in scheduled:
            scheduled[ru] *= self.services_by_ru[ru]

        if not scheduled:
            raise ValueError("Scheduled resources list cannot be empty.")
        if len(scheduled) != len(capacities):
            raise ValueError("Resources and capacities must have the same length.")

        ratios = {ru: scheduled[ru] / capacities[ru] for ru in capacities}
        n = len(ratios)
        sum_ratios = sum(ratios.values())
        sum_squares = sum(x ** 2 for x in ratios.values())
        if sum_squares == 0:
            return 0.0, ratios
        fairness = (sum_ratios ** 2) / (n * sum_squares)
        return fairness, ratios

    def gini_fairness_index(
            self, bool_scheduled: List[bool], capacities: Mapping[Any, float]
    ) -> Tuple[float, Mapping[Any, float]]:
        """
        Calcula una medida de equidad basada en el coeficiente de Gini aplicado
        a los ratios entre el recurso asignado y la capacidad (cuota) de cada RU.

        Args:
            bool_scheduled: Lista de booleanos que indica qué servicios están programados.
            capacities: Mapeo con la capacidad (o cuota) de cada RU.

        Returns:
            Una tupla que contiene:
              - La medida de equidad (float) en el rango [0, 1], donde 1 indica equidad perfecta.
              - Un mapeo de ratios (recurso asignado / capacidad) para cada RU.
        """
        # Construir el diccionario de recursos asignados (scheduled) por RU.
        scheduled = {}
        for service, scheduled_flag in zip(self.revenue.keys(), bool_scheduled):
            ru = self.revenue[service]["ru"]
            # Se suma la "importancia" si el servicio está programado.
            scheduled[ru] = scheduled.get(ru, 0) + (self.revenue[service]["importance"] if scheduled_flag else 0)

        # Multiplicar cada recurso programado por el factor correspondiente de services_by_ru.
        for ru in scheduled:
            scheduled[ru] *= self.services_by_ru[ru]

        if not scheduled:
            raise ValueError("La lista de recursos programados no puede estar vacía.")
        if len(scheduled) != len(capacities):
            raise ValueError("Los RUs de recursos asignados y capacidades deben coincidir en cantidad.")

        # Calcular los ratios: recurso asignado / capacidad
        ratios = {ru: scheduled[ru] / capacities[ru] for ru in capacities}

        # Convertir los valores de ratios a una lista para calcular el Gini.
        values = list(ratios.values())
        n = len(values)
        total = sum(values)
        if total == 0:
            # Si la suma es 0, se puede interpretar como igualdad (ningún RU recibe recurso)
            return 1.0, ratios

        # Calcular el coeficiente de Gini usando la fórmula basada en el ordenamiento:
        #   G = (2 * sum_{i=1}^n (i * x_i_sorted)) / (n * sum(x)) - (n + 1) / n
        sorted_values = sorted(values)
        cumulative = 0
        for i, value in enumerate(sorted_values, start=1):
            cumulative += i * value
        gini = (2 * cumulative) / (n * total) - (n + 1) / n

        # Convertir el coeficiente de desigualdad en una medida de equidad.
        fairness = 1 - gini
        return fairness, ratios

    def atkinson_fairness_index(
            self, bool_scheduled: List[bool], capacities: Mapping[Any, float], epsilon: float = 0.5
    ) -> Tuple[float, Mapping[Any, float]]:
        """
        Calcula una medida de equidad basada en el índice de Atkinson aplicado
        a los ratios entre el recurso asignado y la capacidad (cuota) de cada RU.

        Args:
            bool_scheduled: Lista de booleanos que indica qué servicios están programados.
            capacities: Mapeo con la capacidad (o cuota) de cada RU.
            epsilon: Parámetro de aversión a la desigualdad (usualmente en (0,1]). Por defecto 0.5.

        Returns:
            Una tupla que contiene:
              - La medida de equidad (float) en el rango [0, 1], donde 1 indica equidad perfecta.
              - Un mapeo de ratios (recurso asignado / capacidad) para cada RU.
        """
        # Construir el diccionario de recursos asignados (scheduled) por RU.
        scheduled = {}
        for service, scheduled_flag in zip(self.revenue.keys(), bool_scheduled):
            ru = self.revenue[service]["ru"]
            scheduled[ru] = scheduled.get(ru, 0) + (self.revenue[service]["importance"] if scheduled_flag else 0)

        for ru in scheduled:
            scheduled[ru] *= self.services_by_ru[ru]

        if not scheduled:
            raise ValueError("La lista de recursos programados no puede estar vacía.")
        if len(scheduled) != len(capacities):
            raise ValueError("Los RUs de recursos asignados y capacidades deben coincidir en cantidad.")

        # Calcular los ratios: recurso asignado / capacidad
        ratios = {ru: scheduled[ru] / capacities[ru] for ru in capacities}

        values = list(ratios.values())
        n = len(values)
        if n == 0:
            raise ValueError("No se han calculado ratios para el índice de Atkinson.")
        mean = sum(values) / n
        if mean == 0:
            # Si la media es 0, se puede interpretar como igualdad (ningún RU recibe recurso)
            return 1.0, ratios

        import math

        # Calcular el índice de Atkinson
        if epsilon == 1:
            # Versión logarítmica; se asume que todos los valores son positivos.
            try:
                geo_mean = math.exp(sum(math.log(x) for x in values) / n)
            except ValueError:
                raise ValueError("Todos los valores de ratio deben ser positivos para epsilon = 1 en Atkinson.")
            atkinson_index = 1 - geo_mean / mean
        else:
            sum_power = sum(x ** (1 - epsilon) for x in values)
            term = (sum_power / n) ** (1 / (1 - epsilon))
            atkinson_index = 1 - term / mean

        # Convertir el índice de desigualdad de Atkinson en una medida de equidad.
        fairness = 1 - atkinson_index
        return fairness, ratios

    # === Private Helper Methods ===

    def get_n_services_by_ru(self) -> Mapping[str, int]:
        """
        Count the number of services per RU based on revenue behavior.

        Returns:
            A mapping from RU to service count.
        """
        services_by_ru = {}
        for service, data in self.revenue.items():
            ru = data["ru"]
            services_by_ru[ru] = services_by_ru.get(ru, 0) + 1
        return services_by_ru

    def get_capacities(self) -> Mapping[str, float]:
        """
        Calculate capacities for each RU as a percentage of total services.

        Returns:
            A mapping from RU to capacity percentage.
        """
        return {ru: (count / self.n_services) * 100 for ru, count in self.services_by_ru.items()}

    def get_departure_time_indexer(self) -> Mapping[int, str]:
        """
        Build an index mapping where keys are departure time indices and values are the corresponding service IDs.

        Returns:
            A mapping from integer index to service identifier.
        """
        dt_indexer = {}
        i = 0
        for service, stops in self.requested_schedule.items():
            # Each service provides (number of stops - 1) departure times.
            for _ in range(len(stops) - 1):
                dt_indexer[i] = service
                i += 1
        return dt_indexer

    def _calculate_boundaries(self) -> Boundaries:
        """
        Calculate boundaries for the departure times of each service.

        Returns:
            A Boundaries object containing the real (and empty discrete) boundaries.
        """
        boundaries = []
        for service, stops in self.requested_schedule.items():
            stop_keys = list(stops.keys())
            ot_idx = 0
            for i in range(len(stop_keys) - 1):
                if i == 0:
                    lower_bound = stops[stop_keys[i]][1] - self.im_mod_margin
                    upper_bound = stops[stop_keys[i]][1] + self.im_mod_margin
                else:
                    travel_time = self.operational_times[service][ot_idx]
                    stop_time = self.operational_times[service][ot_idx + 1]
                    ot_idx += 2
                    lower_bound = self.updated_schedule[service][stop_keys[i - 1]][1] + travel_time + stop_time
                    max_dt_original = stops[stop_keys[i]][1] + self.max_stop_time
                    max_dt_updated = lower_bound + (self.max_stop_time - stop_time)
                    upper_bound = min(max_dt_original, max_dt_updated)
                boundaries.append([lower_bound, upper_bound])
        return Boundaries(real=boundaries, discrete=[])

    def _departure_time_feasibility(self, S_i: np.array) -> bool:
        """
        Check whether the departure times in the solution are conflict‐free.

        Args:
            S_i: A boolean array of scheduling decisions.

        Returns:
            True if no conflicts exist; otherwise, False.
        """
        S_i_bool = np.array(S_i, dtype=bool)
        return not np.any((S_i_bool * self.conflict_matrix)[S_i_bool])

    def _feasible_boundaries(self, solution: Solution) -> bool:
        """
        Check that each real variable in the solution lies within its corresponding boundary.

        Args:
            solution: A Solution object containing real departure times.

        Returns:
            True if all values are within bounds; otherwise, False.
        """
        return all(
            self.boundaries.real[i][0] <= rv <= self.boundaries.real[i][1]
            for i, rv in enumerate(solution.real)
        )

    def _get_conflict_matrix(self) -> np.array:
        """
        Compute the conflict matrix among services based on the updated schedule.

        Returns:
            A boolean numpy array where each entry [i, j] indicates whether service i and service j conflict.
        """

        @cache
        def get_x_line_equation(A, B):
            x_coords = (A[0], B[0])
            y_coords = (A[1], B[1])
            m = (y_coords[1] - y_coords[0]) / (x_coords[1] - x_coords[0])
            c = y_coords[0] - m * x_coords[0]
            return lambda y: (y - c) / m

        def infer_times(service: str, station: str, origin: bool = True) -> float:
            idx = 1 if origin else 0
            if station in self.updated_schedule[service]:
                return self.updated_schedule[service][station][idx]

            stations = list(self.updated_schedule[service].keys())
            station_pos = self.line_stations[station]
            before, after = None, None
            stations_pos = [self.line_stations[s] for s in stations]
            for i in range(len(stations) - 1):
                if stations_pos[i] < station_pos < stations_pos[i + 1]:
                    before = stations[i]
                    after = stations[i + 1]
                    break
            if before is None or after is None:
                raise ValueError(f"Station {station} not found in service {service}")
            A = (self.updated_schedule[service][before][1], self.line_stations[before])
            B = (self.updated_schedule[service][after][0], self.line_stations[after])
            line_eq = get_x_line_equation(A, B)
            return line_eq(self.line_stations[station])

        n = len(self.requested_schedule)
        conflict_matrix = np.zeros((n, n), dtype=bool)
        service_keys = list(self.requested_schedule.keys())

        for i, service in enumerate(service_keys):
            stop_keys = list(self.requested_schedule[service].keys())
            for k in range(len(stop_keys) - 1):
                departure_station = stop_keys[k]
                arrival_station = stop_keys[k + 1]
                departure_time = self.updated_schedule[service][departure_station][1]
                arrival_time = self.updated_schedule[service][arrival_station][0]

                for j, other_service in enumerate(service_keys):
                    if other_service == service or conflict_matrix[i, j]:
                        continue

                    other_first_departure = list(self.requested_schedule[other_service].values())[0][1]
                    if other_first_departure > arrival_time:
                        continue

                    other_stop_keys = list(self.requested_schedule[other_service].keys())
                    stations_between = [
                        s for s in other_stop_keys
                        if self.line_stations[departure_station] <= self.line_stations[s] <= self.line_stations[
                            arrival_station]
                    ]
                    if not stations_between:
                        continue

                    trips = set()
                    for s in stations_between:
                        idx = other_stop_keys.index(s)
                        if 0 < idx < len(other_stop_keys) - 1:
                            trips.add((other_stop_keys[idx - 1], s))
                            trips.add((s, other_stop_keys[idx + 1]))
                        elif idx == 0:
                            trips.add((s, other_stop_keys[idx + 1]))
                        elif idx == len(other_stop_keys) - 1:
                            trips.add((other_stop_keys[idx - 1], s))

                    for trip in trips:
                        other_start, other_end = trip
                        inference_departure_station = max(
                            [departure_station, other_start], key=lambda x: self.line_stations[x]
                        )
                        inference_arrival_station = min(
                            [arrival_station, other_end], key=lambda x: self.line_stations[x]
                        )
                        other_departure_time = infer_times(other_service, inference_departure_station, origin=True)
                        other_arrival_time = infer_times(other_service, inference_arrival_station, origin=False)
                        original_departure_time = infer_times(service, inference_departure_station, origin=True)
                        original_arrival_time = infer_times(service, inference_arrival_station, origin=False)
                        dt_gap = other_departure_time - original_departure_time
                        at_gap = other_arrival_time - original_arrival_time
                        same_sign = dt_gap * at_gap > 0
                        if same_sign and all(abs(t) >= 2 * self.safe_headway for t in (dt_gap, at_gap)):
                            continue
                        else:
                            conflict_matrix[i, j] = True
                            conflict_matrix[j, i] = True
        return conflict_matrix

    def _travel_times_feasibility(self, S_i: np.array) -> bool:
        """
        Check whether the travel times in the updated schedule are feasible.

        Args:
            S_i: A boolean array of scheduling decisions.

        Returns:
            True if travel times are feasible; otherwise, False.
        """
        for i, service in enumerate(self.requested_schedule):
            if not S_i[i]:
                continue
            if not self.service_is_feasible(service):
                return False
        return True

    def _update_dynamic_bounds(
            self, j: int, ot_idx: int, proposed_times: List[float], updated_boundaries: List[Tuple[float, float]]
    ) -> Tuple[int, List[Tuple[float, float]]]:
        """
        Update dynamic boundaries for departure times based on operational times.

        Args:
            j: Current index in the requested times.
            ot_idx: Current index in the operational times.
            proposed_times: Proposed departure times.
            updated_boundaries: Current boundaries list.

        Returns:
            A tuple of updated ot_idx and boundaries.
        """
        if j != len(self.requested_times) - 1 and self.dt_indexer[j + 1] == self.dt_indexer[j]:
            travel_time = self.operational_times[self.dt_indexer[j]][ot_idx]
            stop_time = self.operational_times[self.dt_indexer[j]][ot_idx + 1]
            ot_idx += 2
            lower_bound = proposed_times[j] + travel_time + stop_time
            max_dt_original = self.requested_times[j + 1] + self.im_mod_margin
            max_dt_updated = lower_bound + (self.max_stop_time - stop_time)
            upper_bound = min(max_dt_original, max_dt_updated)
            updated_boundaries[j + 1] = (lower_bound, upper_bound)
        else:
            ot_idx = 0
        return ot_idx, updated_boundaries

    # === Static and Cached Methods ===

    @staticmethod
    def penalty_function(x: float, k: int) -> float:
        """
        Compute the penalty based on a normalized deviation.

        Args:
            x: Normalized deviation.
            k: Scaling factor.

        Returns:
            Penalty value (float).
        """
        return 1 - e ** (-k * x ** 2) * (0.5 * cos(pi * x) + 0.5)

    @cache
    def truth_table(self, dim: int) -> List[List[int]]:
        """
        Generate a truth table (all binary combinations) for the given dimension.

        Args:
            dim: Dimension of the truth table.

        Returns:
            A list of binary combinations.
        """
        if dim < 1:
            return [[]]
        sub_tt = self.truth_table(dim - 1)
        return [row + [val] for row in sub_tt for val in [0, 1]]
