"""Utils for benchmarks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typing import Mapping, Tuple

from geopy.distance import geodesic
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pathlib import Path
from robin.supply.entities import Line, Service, Supply
from shapely.geometry.polygon import LinearRing, Polygon
from descartes import PolygonPatch
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
        fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)


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

    def get_default_color_cycle(self):
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_list = list(color_cycle)
        return color_list

    def minutes_to_hhmm(self, minutes: int, pos) -> str:
        hours = int(minutes // 60)
        minutes = int(minutes % 60)
        hours = str(hours).zfill(2)
        minutes = str(minutes).zfill(2)
        label = f'{hours}:{minutes} h.'
        return label

    def round_to_nearest_half_hour(self, minutes, round_down=True):
        hours = minutes // 60
        remainder_minutes = minutes % 60

        if round_down:
            if remainder_minutes < 30:
                rounded_minutes = 0
            else:
                rounded_minutes = 30
        else:
            if remainder_minutes >= 30:
                rounded_minutes = 30
            else:
                hours += 1
                rounded_minutes = 0

        total_rounded_minutes = hours * 60 + rounded_minutes
        return total_rounded_minutes

    def plot(self,
             main_title: str = "Marey Diagram",
             plot_security_gaps: bool = False,
             security_gap: int = 10,
             save_path: Union[Path, None] = None
             ) -> None:
        color_list = self.get_default_color_cycle()

        fig, ax = plt.subplots(figsize=(15, 8))

        min_x = 24 * 60
        max_x = 0
        color_idx = 0
        for train_id, stations in self.schedule_data.items():
            times = [time for station, (arrival, departure) in stations.items() for time in (arrival, departure)]
            if min(times) < min_x:
                min_x = min(times)
            if max(times) > max_x:
                max_x = max(times)
            station_indices = [self.station_positions[station] for station in stations.keys() for _ in range(2)]
            ax.plot(times,
                    station_indices,
                    color=color_list[color_idx],
                    marker='o',
                    label=train_id)

            if plot_security_gaps:
                stops = list(stations.keys())
                for i in range(len(stops) - 1):
                    departure_x = stations[stops[i]][1]
                    arrival_x = stations[stops[i + 1]][0]
                    if departure_x < min_x:
                        min_x = departure_x
                    if arrival_x > max_x:
                        max_x = arrival_x
                    departure_station_y = self.station_positions[stops[i]]
                    arrival_station_y = self.station_positions[stops[i + 1]]
                    gap = security_gap // 2
                    vertices = [(departure_x - gap, departure_station_y), (arrival_x - gap, arrival_station_y),
                                (arrival_x + gap, arrival_station_y), (departure_x + gap, departure_station_y)]
                    ring_mixed = Polygon(vertices)
                    ring_patch = PolygonPatch(ring_mixed,
                                              facecolor=color_list[color_idx],
                                              edgecolor=color_list[color_idx],
                                              alpha=0.6)
                    ax.add_patch(ring_patch)
            color_idx += 1

        for spn in ('top', 'right', 'bottom', 'left'):
            ax.spines[spn].set_visible(True)
            ax.spines[spn].set_linewidth(1.0)
            ax.spines[spn].set_color('#A9A9A9')

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_yticks(tuple(self.station_positions.values()))
        ax.set_yticklabels(self.station_positions.keys(), fontsize=14)

        ax.grid(True)
        ax.grid(True, color='#A9A9A9', alpha=0.3, zorder=1, linestyle='-', linewidth=1.0)
        ax.set_xlim(self.round_to_nearest_half_hour(min_x - 10),
                    self.round_to_nearest_half_hour(max_x + 10, round_down=False))
        ax.set_title(main_title, fontweight='bold', fontsize=20)
        ax.set_xlabel('Tiempo (HH:MM)', fontsize=18)
        ax.set_ylabel('Estaciones', fontsize=18)

        ax.xaxis.set_major_locator(MultipleLocator(30))
        formatter = FuncFormatter(self.minutes_to_hhmm)
        ax.xaxis.set_major_formatter(formatter)

        plt.tight_layout()
        plt.show()

        if save_path:
            fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)


def infer_line_stations(lines: List[Line]) -> Mapping[str, Tuple[float, float]]:
    """
    Get list of stations that are part of the corridor

    Returns:
        corridor_stations (List[str]): list of strings with the station ids
    """
    # Initialize corridor with max length trip
    max_len_line = lines.pop(lines.index(max(lines, key=lambda x: len(x.stations))))
    line_stations = list([sta for sta in max_len_line.stations])

    # Complete line with other stations that are not in the initial line
    for line in lines:
        for i, station in enumerate(line.stations):
            if station not in line_stations:
                line_stations.insert(line_stations.index(line.stations[i + 1]), station)

    return {station.id: station.coords for station in line_stations}


def get_services_by_tsp_df(services: List[Service]) -> pd.DataFrame:
    """
    Get the number of services by TSP.

    Args:
        services: List of services.

    Returns:
        pd.DataFrame: DataFrame with the number of services by TSP.
    """
    services = {service.id: service for service in services}
    services_by_tsp = {}
    for service in services:
        if services[service].tsp.name not in services_by_tsp:
            services_by_tsp[services[service].tsp.name] = 1
        else:
            services_by_tsp[services[service].tsp.name] += 1

    return pd.DataFrame.from_dict(services_by_tsp, orient='index', columns=['Number of Services'])
