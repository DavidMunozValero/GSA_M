"""Utils for benchmarks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typing import Mapping, Tuple

from geopy.distance import geodesic
from matplotlib.colors import ListedColormap
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


def sns_box_plot(df: pd.DataFrame,
                 x_data: str,
                 y_data: str,
                 title: str,
                 x_label: str,
                 y_label: str,
                 hue: Union[str, None] = None,
                 save_path: Union[Path, None] = None,
                 fig_size: tuple = (10, 6)
                 ) -> None:
    fig, ax = plt.subplots(figsize=fig_size)

    ax.set_title(title, fontweight='bold', fontsize=18)

    # Draw the boxplot and stripplot
    boxplot = sns.boxplot(data=df, x=x_data, y=y_data, hue=hue, dodge=True, zorder=1, boxprops=dict(alpha=.3), ax=ax)
    stripplot = sns.stripplot(data=df, x=x_data, y=y_data, hue=hue, dodge=True, alpha=0.5, zorder=1, ax=ax)

    # Remove the stripplot legend handles
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [handle for handle, label in zip(handles, labels) if 'line' not in str(type(handle))]

    if hue:
        ax.legend(handles=new_handles, title=hue, fontsize=12, title_fontsize=14)

    ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    for spn in ('top', 'right', 'bottom', 'left'):
        ax.spines[spn].set_visible(True)
        ax.spines[spn].set_linewidth(1.0)
        ax.spines[spn].set_color('#A9A9A9')

    plt.show()
    if save_path:
        fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)


def sns_line_plot(df: pd.DataFrame,
                  x_data: str,
                  y_data: str,
                  title: str,
                  x_label: str,
                  y_label: str,
                  hue: Union[str, None] = None,
                  save_path: Union[Path, None] = None,
                  legend_type: str = "outside",
                  x_limit: tuple = (-1, 100),
                  y_limit: tuple = (-1, 4000),
                  fig_size: tuple = (10, 6)
                  ) -> None:
    fig, ax = plt.subplots(figsize=fig_size)

    ax.set_title(title, fontweight='bold', fontsize=18)
    ax.set_xlim(x_limit)
    ax.set_ylim(y_limit)

    sns.lineplot(ax=ax,
                 data=df,
                 x=x_data,
                 y=y_data,
                 hue=hue,
                 legend=True)

    ax.grid(axis='y', color='#A9A9A9', alpha=0.3, zorder=1)

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)

    for spn in ('top', 'right', 'bottom', 'left'):
        ax.spines[spn].set_visible(True)
        ax.spines[spn].set_linewidth(1.0)
        ax.spines[spn].set_color('#A9A9A9')

    if legend_type == 'outside':
        plt.legend(
            loc='upper center',  # Base de la posición (arriba y centrada)
            bbox_to_anchor=(0.5, -0.2),  # Desplazamiento debajo del área de la gráfica
            ncol=2,  # Organiza la leyenda en dos columnas
            frameon=True,  # Muestra el marco de la leyenda (opcional)
        )
    else:
        # Inside, bottom right
        plt.legend(loc='lower right')

    plt.tight_layout(rect=[0, 0.15, 1, 1])  # Ajusta los márgenes para incluir la leyenda debajo

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


def get_schedule_from_supply(path: Union[Path, None] = None,
                             supply: Union[Supply, None] = None
                             ) -> Mapping[str, Mapping[str, List[int]]]:
    if not supply:
        supply = Supply.from_yaml(path=path)
    requested_schedule = {}
    for service in supply.services:
        requested_schedule[service.id] = {}
        time = service.time_slot.start
        delta = time.total_seconds() // 60
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

    def plot(self,
             main_title: str = "Diagrama de Marey",
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
                    color=color_list[color_idx % len(color_list)],
                    marker='o',
                    linewidth=2.0,
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
                    gap = security_gap
                    vertices = [(departure_x - gap, departure_station_y), (arrival_x - gap, arrival_station_y),
                                (arrival_x + gap, arrival_station_y), (departure_x + gap, departure_station_y)]
                    ring_mixed = Polygon(vertices)
                    ring_patch = PolygonPatch(ring_mixed,
                                              facecolor=color_list[color_idx % len(color_list)],
                                              edgecolor=color_list[color_idx % len(color_list)],
                                              alpha=0.6)
                    ax.add_patch(ring_patch)
            color_idx += 1

        for spn in ('top', 'right', 'bottom', 'left'):
            ax.spines[spn].set_visible(True)
            ax.spines[spn].set_linewidth(1.0)
            ax.spines[spn].set_color('#A9A9A9')

        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_yticks(tuple(self.station_positions.values()))
        ax.set_yticklabels(self.station_positions.keys(), fontsize=20)

        ax.grid(True)
        ax.grid(True, color='#A9A9A9', alpha=0.3, zorder=1, linestyle='-', linewidth=1.0)
        ax.set_xlim(round_to_nearest_half_hour(min_x - 10),
                    round_to_nearest_half_hour(max_x + 10, round_down=False))
        ax.set_title(main_title, fontweight='bold', fontsize=30)
        ax.set_xlabel('Hora (HH:MM)', fontsize=24)
        ax.set_ylabel('Estaciones', fontsize=24)

        ax.xaxis.set_major_locator(MultipleLocator(90))
        formatter = FuncFormatter(minutes_to_hhmm)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.get_xticklabels(), rotation=70, horizontalalignment='right', fontsize=20)

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
            if station not in line_stations and i < len(line.stations) - 1:
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

    df = pd.DataFrame.from_dict(services_by_tsp, orient='index', columns=['Number of Services'])
    return df


def minutes_to_hhmm(minutes: int, pos) -> str:
    hours = int(minutes // 60)
    minutes = int(minutes % 60)
    hours = str(hours).zfill(2)
    minutes = str(minutes).zfill(2)
    label = f'{hours}:{minutes} h.'
    return label


def round_to_nearest_half_hour(minutes, round_down=True):
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


def plot_marey_chart(requested_supply: Supply,
                     scheduled_supply: Union[Supply, None] = None,
                     colors_by_tsp: bool = False,
                     main_title: str = "Diagrama de Marey",
                     plot_security_gaps: bool = False,
                     security_gap: int = 10,
                     x_limits: tuple = (60, 26*60),
                     save_path: Union[Path, None] = None
                     ) -> None:
    requested_supply = requested_supply
    # TODO: Check if supply has multiple branches in corridor
    path = requested_supply.corridors[0].paths[0][:-2]
    line = {sta.id: sta.coords for sta in path}
    station_positions = get_stations_positions(line, scale=1000)
    qualitative_colors = sns.color_palette("pastel", 10)
    my_cmap = ListedColormap(sns.color_palette(qualitative_colors).as_hex())

    tsps = sorted(set([service.tsp.name for service in requested_supply.services]))
    services_dict = {service.id: service for service in requested_supply.services}
    tsp_colors = {tsp: my_cmap(i) for i, tsp in enumerate(tsps)}
    service_color = {service.id: tsp_colors[service.tsp.name] for service in requested_supply.services}

    fig, ax = plt.subplots(figsize=(15, 8))

    min_x = 0
    max_x = 24 * 60
    color_idx = 0
    schedule_data = get_schedule_from_supply(supply=requested_supply)
    labels_added = set()
    # Set default color for requested services
    requested_color = '#D3D3D3'
    polygons = []
    for train_id, stations in schedule_data.items():
        times = [time for station, (arrival, departure) in stations.items() for time in (arrival, departure)]
        if min(times) < min_x:
            min_x = min(times)
        if max(times) > max_x:
            max_x = max(times)
        station_indices = [station_positions[station] for station in stations.keys() for _ in range(2)]
        if services_dict[train_id].tsp.name not in labels_added:
            label = services_dict[train_id].tsp.name
            labels_added.add(label)
        else:
            label = None

        ax.plot(times,
                station_indices,
                color=service_color[train_id],
                marker='o',
                linewidth=2.0,
                label=label)

        if plot_security_gaps:
            stops = list(stations.keys())
            for i in range(len(stops) - 1):
                departure_x = stations[stops[i]][1]
                arrival_x = stations[stops[i + 1]][0]
                if departure_x < min_x:
                    min_x = departure_x
                if arrival_x > max_x:
                    max_x = arrival_x
                departure_station_y = station_positions[stops[i]]
                arrival_station_y = station_positions[stops[i + 1]]
                gap = security_gap
                vertices = [(departure_x - gap, departure_station_y), (arrival_x - gap, arrival_station_y),
                            (arrival_x + gap, arrival_station_y), (departure_x + gap, departure_station_y)]
                ring_mixed = Polygon(vertices)
                polygons.append(ring_mixed)
                ring_patch = PolygonPatch(ring_mixed,
                                          facecolor=requested_color,
                                          edgecolor=requested_color,
                                          alpha=0.6)
                ax.add_patch(ring_patch)

        for i, pa in enumerate(polygons):
            for j, pb in enumerate(polygons[i:], start=i):
                if i == j:
                    continue
                intersection = pa.intersection(pb)
                if not intersection.is_empty and intersection.geom_type == 'Polygon':
                    intersection_patch = PolygonPatch(intersection, facecolor='crimson', edgecolor='crimson', alpha=0.5)
                    ax.add_patch(intersection_patch)

    for spn in ('top', 'right', 'bottom', 'left'):
        ax.spines[spn].set_visible(True)
        ax.spines[spn].set_linewidth(1.0)
        ax.spines[spn].set_color('#A9A9A9')

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_yticks(tuple(station_positions.values()))
    ax.set_yticklabels(station_positions.keys(), fontsize=16)

    ax.grid(True)
    ax.grid(True, color='#A9A9A9', alpha=0.3, zorder=1, linestyle='-', linewidth=1.0)
    # ax.set_xlim(round_to_nearest_half_hour(min_x - 10), round_to_nearest_half_hour(max_x + 10, round_down=False))
    ax.set_xlim(x_limits)
    ax.set_title(main_title, fontweight='bold', fontsize=24)
    ax.set_xlabel('Time (HH:MM)', fontsize=18)
    ax.set_ylabel('Stations', fontsize=18)

    ax.legend()
    ax.xaxis.set_major_locator(MultipleLocator(60))
    formatter = FuncFormatter(minutes_to_hhmm)
    ax.xaxis.set_major_formatter(formatter)
    plt.setp(ax.get_xticklabels(), rotation=70, horizontalalignment='right', fontsize=20)

    plt.tight_layout()
    plt.show()

    if save_path:
        fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)
