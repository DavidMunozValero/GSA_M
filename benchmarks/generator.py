"""Generator of railway service requests."""

import numpy as np
from scipy.stats import loguniform

from typing import List, Mapping, Union


def get_lines(corridor: Mapping[str, Mapping],
              path: Union[List, None] = None
              ) -> List[List[str]]:
    """
    Get all the lines in the corridor

    Args:
        corridor (dict): dictionary with the corridor structure
        path (list, optional): list of nodes

    Returns:
        list of lines
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


def sample_line(lines: list) -> list:
    """
    Sample a random line from the list of lines

    Args:
        lines (list): list of lines

    Returns:
        list: random line
    """
    return lines[np.random.randint(len(lines))]


def sample_route(line: list) -> list:
    """
    Sample a random route from line

    Args:
        line (list): list of stations

    Returns:
        list: random route
    """
    return line[np.random.randint(0, len(line) - 1):]


def get_timetable(route: list) -> dict:
    """
    Generate random timetable for route r

    Args:
        route (list): list of stations

    Returns:
        dict: timetable
    """
    timetable = {}
    arrival_time = np.random.randint(0, 24 * 60)
    departure_time = arrival_time
    for i, sta in enumerate(route):
        if i == 0 or i == len(route) - 1:
            timetable[sta] = (arrival_time, arrival_time)
        else:
            timetable[sta] = (arrival_time, departure_time)

        arrival_time += np.random.randint(30, 120)
        departure_time = arrival_time + np.random.randint(2, 8)

    return timetable


def get_schedule_request(corridor: Mapping[str, Mapping],
                         n_services: int = 1
                         ) -> Mapping[int, Mapping]:
    """
    Generate random timetable

    Args:
        corridor (Mapping[str, Mapping]): corridor structure.
        n_services (int): number of services

    Returns:
        Mapping[int, Mapping]: timetable
    """
    lines = get_lines(corridor)
    return {i: get_timetable(sample_route(sample_line(lines))) for i in range(1, n_services + 1)}


def get_revenue_behaviour(schedule: dict) -> dict:
    """
    Get revenue behaviour

    Args:
        schedule (dict): schedule

    Returns:
        dict: revenue behaviour
    """
    revenue = {}
    bias = [0.2, 0.35, 0.1]
    for service in schedule:
        b = np.random.choice(bias)
        base_price = 55 * len(schedule[service])
        canon = base_price + b * base_price
        k = np.round(loguniform.rvs(0.01, 100, 1), 2)
        max_penalty = canon * 0.4
        dt_penalty = np.round(max_penalty * 0.35, 2)
        tt_penalty = np.round((max_penalty - dt_penalty) / (len(schedule[service]) - 1), 2)
        revenue[service] = {'canon': canon, 'k': k, 'dt_max_penalty': dt_penalty, 'tt_max_penalty': tt_penalty}
    return revenue
