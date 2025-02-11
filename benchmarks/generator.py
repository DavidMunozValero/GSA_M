"""Generator of railway service requests."""

import numpy as np

from collections import defaultdict
from geopy.distance import geodesic
from scipy.stats import loguniform
from typing import Dict, List, Mapping, Tuple, Union

from robin.supply.entities import Supply


def get_lines(corridor: Mapping[str, Mapping],
              path: Union[List, None] = None
              ) -> List[List[str]]:
    """
    Get all the lines in the corridor.

    Args:
        corridor (Mapping[str, Mapping]): corridor structure.
        path (Union[List, None]): path.

    Returns:
        List[List[str]]: list of lines
    """
    if path is None:
        path = []

    lines = []
    for node, child in corridor.items():
        new_path = path + [node]
        if not child:  # If the node has no children, it is a leaf
            lines.append(new_path)
        else:
            lines.extend(get_lines(child, new_path))  # If the node has children, we call the function recursively

    return lines


def sample_line(lines: List[List[str]]) -> List[str]:
    """
    Sample a random line from the list of lines.

    Args:
        lines (List[List[str]]): list of lines.

    Returns:
        List[str]: random line.
    """
    return lines[np.random.randint(len(lines))]


def sample_route(line: List[str]) -> List[str]:
    """
    Sample a random route from line.

    Args:
        line (List[str]): line.

    Returns:
        List[str]: random route.
    """
    return line[np.random.randint(0, len(line) - 1):]


def get_timetable(route: List[str]) -> Mapping[str, List[float]]:
    """
    Generate random timetable for given route.

    Args:
        route (List[str]): route.

    Returns:
        Mapping[str, List[float]]: timetable.
    """
    timetable = {}
    arrival_time = np.random.randint(0, 24 * 60)
    departure_time = arrival_time
    for i, sta in enumerate(route):
        if i == 0 or i == len(route) - 1:
            timetable[sta] = [arrival_time, arrival_time]
        else:
            timetable[sta] = [arrival_time, departure_time]

        arrival_time += np.random.randint(30, 120)
        departure_time = arrival_time + np.random.randint(2, 8)

    return timetable


def get_schedule_request(corridor: Mapping[str, Mapping],
                         n_services: int = 1
                         ) -> Mapping[int, Mapping[str, List[float]]]:
    """
    Generate random timetable.

    Args:
        corridor (Mapping[str, Mapping]): corridor structure.
        n_services (int): number of services.

    Returns:
        Mapping[int, Mapping[str, List[float]]]: schedule request.
    """
    lines = get_lines(corridor)
    schedule_request = {}
    for i in range(1, n_services + 1):
        schedule_request[i] = get_timetable(sample_route(sample_line(lines)))
    return schedule_request


def get_revenue_behavior(supply: Supply,
                          alpha: float = 2/3) -> Mapping[str, Mapping[str, float]]:
    """
    Get revenue behavior

    Args:
        supply (Supply): supply.
        alpha (float): alpha to adjust canon by 100 passengers.

    Returns:
        Mapping[int, Mapping[str, float]]: revenue behavior.
    """
    revenue = {}
    tsp_k = {}
    for service in supply.services:
        sta_coords = [sta.coords for sta in service.line.stations]
        distances = [geodesic(sta_coords[i], sta_coords[i + 1]).km for i in range(len(sta_coords) - 1)]
        distance_factor = 7 * sum(distances)
        service_capacity = service.rolling_stock.total_capacity
        capacity_factor = (alpha * service_capacity) / 100 * 1.67
        stations_factor = 18 + (len(sta_coords) - 2) * 65 + 165
        total_canon = (distance_factor + capacity_factor + stations_factor) / 100
        max_penalty = total_canon * 0.3
        dt_penalty = np.round(max_penalty * 0.35, 2)
        tt_penalty = np.round((max_penalty - dt_penalty) / (len(sta_coords) - 1), 2)
        if service.tsp.id not in tsp_k:
            k = np.round(loguniform.rvs(0.01, 100, 1), 2)
            tsp_k[service.tsp.id] = k
        else:
            k = tsp_k[service.tsp.id]
        revenue[service.id] = {'canon': total_canon, 'k': k, 'dt_max_penalty': dt_penalty, 'tt_max_penalty': tt_penalty}
    return revenue


def get_revenue_behavior_deprecated(supply: Supply) -> Mapping[str, Dict[str, float]]:
    """
    Calculate revenue behavior parameters for each service in the supply.

    For every service in the supply, this function computes:
      - canon: The base revenue increased by a randomly selected bias factor.
      - k: A random scaling factor drawn from a log-uniform distribution.
      - dt_max_penalty: A penalty value derived from the canon.
      - tt_max_penalty: A penalty value distributed across the service's stations.
      - importance: A normalized random weight assigned to the service within its RU group.

    The services are first grouped by their associated RU (Transport Service Provider).
    Then, for each group, a set of random values is generated and normalized so that the
    sum of importance values in each group is 1. Finally, these importance values are assigned
    to the corresponding services.

    Args:
        supply (Supply): An object containing a list of services. Each service is expected to have:
            - service.id: A unique identifier for the service.
            - service.line.stations: A collection (e.g., list) of stations.
            - service.tsp.id: The identifier of the associated RU.

    Returns:
        Mapping[str, Dict[str, float]]:
            A dictionary mapping each service's ID to a dictionary of computed revenue parameters.
            Each dictionary contains:
                - 'canon': Computed base revenue (float).
                - 'ru': The RU identifier (same as service.tsp.id).
                - 'k': A random scaling factor (float).
                - 'dt_max_penalty': Penalty value for DT (float).
                - 'tt_max_penalty': Penalty value for TT (float).
                - 'importance': Normalized importance weight (float) within its RU group.
    """
    # Predefined bias values for revenue calculation
    bias_options = [0.2, 0.35, 0.1]

    # Dictionaries to hold the computed revenue parameters and group services by RU.
    revenue_by_service: Dict[str, Dict[str, float]] = {}
    services_by_ru: Dict[str, list] = defaultdict(list)

    # Compute revenue parameters for each service and group by RU.
    for service in supply.services:
        # Randomly select a bias factor
        bias = np.random.choice(bias_options)

        # Compute the base price using the number of stations
        num_stations = len(service.line.stations)
        base_price = 55 * num_stations

        # Compute the canon as the base price increased by the bias factor
        canon = base_price * (1 + bias)

        # Generate a random scaling factor 'k' using a log-uniform distribution
        k = float(np.round(loguniform.rvs(0.01, 100, size=1), 2))

        # Calculate penalties based on the canon
        max_penalty = canon * 0.4
        dt_max_penalty = float(np.round(max_penalty * 0.35, 2))
        # Avoid division by zero if there is only one station
        tt_max_penalty = (
            float(np.round((max_penalty - dt_max_penalty) / (num_stations - 1), 2))
            if num_stations > 1 else 0.0
        )

        # Identify the RU for the service
        ru_id = service.tsp.id

        # Store computed revenue parameters for this service
        revenue_by_service[service.id] = {
            'canon': canon,
            'ru': ru_id,
            'k': k,
            'dt_max_penalty': dt_max_penalty,
            'tt_max_penalty': tt_max_penalty,
        }

        # Group the service IDs by their RU
        services_by_ru[ru_id].append(service.id)

    # For each RU group, generate normalized random importance values and assign them to services.
    for ru_id, service_ids in services_by_ru.items():
        num_services = len(service_ids)
        # Generate random values and normalize them so they sum to 1
        random_values = np.random.random(num_services)
        normalized_importance = random_values / random_values.sum()

        for idx, service_id in enumerate(service_ids):
            revenue_by_service[service_id]['importance'] = float(normalized_importance[idx])

    return revenue_by_service


def get_revenue_behavior_old(services: Mapping,
                              alpha: float = 2/3) -> Mapping[str, Mapping[str, float]]:
    """
    Get revenue behavior

    Args:
        supply (Supply): supply.
        alpha (float): alpha to adjust canon by 100 passengers.

    Returns:
        Mapping[int, Mapping[str, float]]: revenue behavior.
    """
    revenue = {}
    for service in services:
        distances = [50 for _ in range(len(services[service]) - 1)]
        distance_factor = 7 * sum(distances)
        service_capacity = 300
        capacity_factor = alpha * service_capacity
        stations_factor = 180 + (len(distances) - 2) * 65 + 165
        total_canon = distance_factor + capacity_factor + stations_factor
        max_penalty = total_canon * 0.4
        dt_penalty = np.round(max_penalty * 0.35, 2)
        tt_penalty = np.round((max_penalty - dt_penalty) / len(distances), 2)
        k = np.round(loguniform.rvs(0.01, 100, 1), 2)
        revenue[service] = {'canon': total_canon, 'k': k, 'dt_max_penalty': dt_penalty, 'tt_max_penalty': tt_penalty}
    return revenue