"""Generator of railway service requests."""

import numpy as np

from geopy.distance import geodesic
from robin.supply.entities import Supply
from scipy.stats import loguniform
from typing import List, Mapping, Tuple, Union


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


def get_revenue_behaviour(supply: Supply,
                          alpha: float = 2/3) -> Mapping[str, Mapping[str, float]]:
    """
    Get revenue behaviour

    Args:
        supply (Supply): supply.
        alpha (float): alpha to adjust canon by 100 passengers.

    Returns:
        Mapping[int, Mapping[str, float]]: revenue behaviour.
    """
    revenue = {}
    tsp_k = {}
    for service in supply.services:
        sta_coords = [sta.coords for sta in service.line.stations]
        distances = [geodesic(sta_coords[i], sta_coords[i+1]).km for i in range(len(sta_coords) - 1)]
        distance_factor = 7 * sum(distances)
        service_capacity = service.rolling_stock.total_capacity
        capacity_factor = alpha * service_capacity
        stations_factor = 180 + (len(sta_coords) - 2) * 65 + 165
        total_canon = distance_factor + capacity_factor + stations_factor
        max_penalty = total_canon * 0.4
        dt_penalty = np.round(max_penalty * 0.35, 2)
        tt_penalty = np.round((max_penalty - dt_penalty) / (len(sta_coords) - 1), 2)
        if service.tsp.id not in tsp_k:
            k = np.round(loguniform.rvs(0.01, 100, 1), 2)
            tsp_k[service.tsp.id] = k
        else:
            k = tsp_k[service.tsp.id]
        revenue[service.id] = {'canon': total_canon, 'k': k, 'dt_max_penalty': dt_penalty, 'tt_max_penalty': tt_penalty}
    return revenue


def get_revenue_behaviour_deprecated(supply: Supply) -> Mapping[str, Mapping[str, float]]:
    """
    Get revenue behaviour

    Args:
        supply (Supply): supply.

    Returns:
        Mapping[int, Mapping[str, float]]: revenue behaviour.
    """
    revenue = {}
    bias = [0.2, 0.35, 0.1]
    for service in supply.services:
        b = np.random.choice(bias)
        base_price = 55 * len(service.line.stations)
        canon = base_price + b * base_price
        k = np.round(loguniform.rvs(0.01, 100, 1), 2)
        max_penalty = canon * 0.4
        dt_penalty = np.round(max_penalty * 0.35, 2)
        tt_penalty = np.round((max_penalty - dt_penalty) / (len(service.line.stations) - 1), 2)
        revenue[service.id] = {'canon': canon, 'k': k, 'dt_max_penalty': dt_penalty, 'tt_max_penalty': tt_penalty}
    return revenue


def get_revenue_behaviour_old(services: Mapping,
                              alpha: float = 2/3) -> Mapping[str, Mapping[str, float]]:
    """
    Get revenue behaviour

    Args:
        supply (Supply): supply.
        alpha (float): alpha to adjust canon by 100 passengers.

    Returns:
        Mapping[int, Mapping[str, float]]: revenue behaviour.
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