"""Utils for benchmarks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typing import Mapping, Tuple

from geopy.distance import geodesic
from pathlib import Path
from robin.supply.entities import Supply
from typing import List, Union


def get_rus_revenue(supply: Supply,
                    df: pd.DataFrame
                    ) -> Mapping[str, float]:
    """
    Get the revenue of each RU.

    Args:
        supply: Supply object.
        df: DataFrame from Robins's output data with columns 'service' and 'price'.

    Returns:
        Mapping[str, float]: RU's revenue.
    """
    services_tsp = {service.id: service.tsp.name for service in supply.services}
    df['tsp'] = df['service'].apply(lambda service_id: services_tsp.get(service_id, np.NaN))
    tsp_revenue = df.groupby('tsp').agg({'price': 'sum'}).to_dict()['price']
    return tsp_revenue


def is_better_solution(rus_revenue: Mapping[str, float],
                       best_solution: Mapping[str, float]
                       ) -> bool:
    """
    Check if the current solution is better than the best solution.

    Args:
        rus_revenue: Revenue of each RU.
        best_solution: Best solution found so far.

    Returns:
        bool: True if the current solution is better than the best solution, False otherwise.
    """
    if not best_solution:
        return True
    elif len(rus_revenue) > len(best_solution):
        return True
    elif sum([rus_revenue[tsp] > best_solution.get(tsp, -np.inf) for tsp in rus_revenue]) >= len(rus_revenue) // 2:
        return True
    return False


def sns_line_plot(df: pd.DataFrame,
                  x_data: str,
                  y_data: str,
                  title: str,
                  x_label: str,
                  y_label: str,
                  save_path: Union[Path, None] = None,
                  fig_size: tuple = (10, 6)
                  ) -> None:
    fig, ax = plt.subplots(figsize=fig_size)

    ax.set_title(title, fontweight='bold')
    # ax.set_xlim(min(df[x_data]), max(df[x_data]))
    # ax.set_ylim(min(df[y_data]), max(df[y_data]))

    sns.lineplot(ax=ax,
                 data=df,
                 x=x_data,
                 y=y_data,
                 legend=True)

    ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.show()
    if save_path:
        fig.savefig(save_path, format='svg', dpi=300, bbox_inches='tight', transparent=True)


def int_input(prompt: str) -> int:
    """
    Get an integer input from the user.

    Args:
        prompt: Message to show to the user.

    Returns:
        int: Integer input.
    """
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print('Invalid input. Please, enter an integer.')


def get_schedule_from_supply(path: Path) -> Mapping[str, Mapping[str, List[int]]]:
    requested_schedule = {}
    supply = Supply.from_yaml(path=path)
    for service in supply.services:
        requested_schedule[service.id] = {}
        time = service.id.split("-")[-1]
        hour, minute = time.split(".")
        delta = int(hour) * 60 + int(minute)
        for stop in service.line.timetable:
            arrival_time = delta + int(service.line.timetable[stop][0])
            departure_time = delta + int(service.line.timetable[stop][1])
            requested_schedule[service.id][stop] = [arrival_time, departure_time]

    return requested_schedule


def get_stations_positions(line, scale: Union[int, None] = None) -> Mapping[str, float]:
    stations_positions = {}
    prev_station = None
    for i, station in enumerate(tuple(line.keys())):
        if i == 0:
            stations_positions[station] = 0
        else:
            prev_distance = tuple(stations_positions.values())[-1]
            stations_distance = geodesic(line[prev_station], line[station]).km
            stations_positions[station] = prev_distance + stations_distance
        prev_station = station

    if not scale:
        return stations_positions

    max_distance = tuple(stations_positions.values())[-1]
    for station in stations_positions:
        stations_positions[station] = np.round(stations_positions[station] / max_distance * 1000, 2)

    return stations_positions


class TrainSchedulePlotter:
    def __init__(self, schedule_data, line: Mapping[str, Tuple[float, float]]):
        self.schedule_data = schedule_data
        self.line = line
        self.station_positions = get_stations_positions(line, scale=1000)

    def plot(self, save_path: Union[Path, None] = None) -> None:
        fig, ax = plt.subplots(figsize=(15, 8))

        for train_id, stations in self.schedule_data.items():
            times = [time for station, (arrival, departure) in stations.items() for time in (arrival, departure)]
            station_indices = [self.station_positions[station] for station in stations.keys() for _ in range(2)]
            ax.plot(times, station_indices, marker='o', label=train_id)

        ax.set_yticks(tuple(self.station_positions.values()))
        ax.set_yticklabels(self.station_positions.keys())

        ax.grid(True)
        ax.set_title('Train schedule', fontweight='bold')
        ax.set_xlabel('Minutes')
        ax.set_ylabel('Stations')
        # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.show()

        if save_path:
            fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=False)