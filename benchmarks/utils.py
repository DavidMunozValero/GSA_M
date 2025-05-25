"""Utils for benchmarks."""

import datetime
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from collections import defaultdict
from descartes import PolygonPatch
from geopy.distance import geodesic
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pathlib import Path
from robin.supply.entities import Station, Line, Corridor, Service, Supply
from shapely.geometry import MultiPolygon, Polygon
from typing import Any, Dict, List, Mapping, Set, Tuple, Union


MARKERS = {
    'departure': {'marker': '^', 'label': 'Departure Station'},
    'arrival': {'marker': 's', 'label': 'Arrival Station'},
    'intermediate': {'marker': 'o', 'label': 'Intermediate Station'}
}

SAFETY_GAP = 10


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

    return {station.id: station.coordinates for station in line_stations}


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

def compute_normalized_positions(
    paths_dict: Mapping[int, Tuple[Station, ...]],
    scale_max: int = 1000
) -> Dict[int, Dict[Station, float]]:
    """
    Compute and normalize cumulative distances for each path.

    Args:
        paths_dict (Mapping[int, Tuple[Station]]): Indexed station sequences.
        scale_max (int): Target maximum for normalized positions.

    Returns:
        Dict[int, Dict[Station, float]]: Station to normalized position mapping per path.
    """
    positions = {}
    for idx, path in paths_dict.items():
        cum = {path[0]: 0.0}
        dist = 0.0
        for a, b in zip(path, path[1:]):
            dist += geodesic(a.coordinates, b.coordinates).kilometers
            cum[b] = dist
        max_dist = max(cum.values()) or 1.0
        positions[idx] = {st: (pos / max_dist) * scale_max for st, pos in cum.items()}
    return positions

def enumerate_unique_paths(
    corridors: Set[Corridor]
) -> Mapping[int, Tuple[Station, ...]]:
    """
    Enumerate and index each unique path across corridors.

    Args:
        corridors (Set[Corridor]): Corridors to extract paths from.

    Returns:
        Dict[int, Tuple[Station, ...]]: Dict mapping path index to unique paths.
    """
    unique = set(tuple(p) for c in corridors for p in c.paths)
    return {i: path for i, path in enumerate(unique)}

def build_graph(tree: Mapping[Station, Mapping], graph: nx.Graph = None) -> nx.Graph:
    """
    Recursively convert a station‐tree into a weighted NetworkX graph.

    Args:
        tree (Mapping[Station, Mapping): corridor tree with stations as keys and branches as values.
        graph (nx.Graph, optional): existing graph to add edges into. Creates new if None.

    Returns:
        nx.Graph: undirected graph with weighted edges representing distances.
    """
    if graph is None:
        graph = nx.Graph()
    for origin, branches in tree.items():
        for dest in branches:
            # Compute distance between station coords
            km = geodesic(origin.coordinates, dest.coordinates).kilometers
            graph.add_edge(origin, dest, weight=km)
            # Recurse into each branch's subtree
            build_graph({dest: branches.get(dest, [])}, graph)
    return graph

def infer_paths(service: Service) -> List[List[Station]]:
    """
    Infers the path of a service based on its line and corridor.

    Args:
        service (Service): The service to infer.

    Returns:
        List[List[Station]]: A list of paths, where each path is a list of Station objects.
    """
    graph = build_graph(service.line.corridor.tree)
    stops = service.line.stations

    if len(stops) < 2:
        return []

    # Get the full path from the first to the last station
    full_path: List[Any] = []
    for origen, destino in zip(stops, stops[1:]):
        seg = nx.shortest_path(graph, origen, destino, weight='weight')
        if full_path:
            full_path.extend(seg[1:])
        else:
            full_path.extend(seg)

    # Detect split points in the path
    split_idxs = sorted({
        idx
        for idx, node in enumerate(full_path)
        if graph.degree[node] != 2
    })

    # Check if the first and last stations are split points
    if 0 not in split_idxs:
        split_idxs.insert(0, 0)
    last = len(full_path) - 1
    if last not in split_idxs:
        split_idxs.append(last)

    # Split the full path into subroutes
    subroutes: List[List[Any]] = []
    for a, b in zip(split_idxs[:-1], split_idxs[1:]):
        subroutes.append(full_path[a: b + 1])

    return subroutes

def get_edges_from_path(path: List[Station]) -> Set[Tuple[Station]]:
    """
    Returns the set of edges for a given path of stations.

    Args:
        path (List[Station]): A list of Station objects representing the path.

    Returns:
        Set[Tuple[Station]]: A set of edges, where each edge is represented as a tuple of two stations.
    """
    edges: Set[Tuple[Station]] = set()
    for i in range(len(path) - 1):
        origin, destination = path[i], path[i + 1]
        edges.add((origin, destination))
    return edges

def shared_edges_between_services(path1: List[Station], path2: List[Station]) -> Set[Tuple[Station]]:
    """
    Checks whether two services (given by their paths) share any track segments.

    Args:
        path1: A list of Station objects for the first service.
        path2: A list of Station objects for the second service.

    Returns:
        A set of shared edges (track segments), if any.
    """
    edges1 = get_edges_from_path(path1)
    edges2 = get_edges_from_path(path2)
    return edges1.intersection(edges2)

def assign_services_to_paths(
     services: List[Service],
    paths_dict: Mapping[int, Tuple[Station, ...]]
) -> Dict[int, List[str]]:
    """
    Determine which services travel along each path by matching edges.

    Args:
        services (List[Service]): Services to classify.
        paths_dict (Mapping[int, Tuple[Station]]): Candidate paths.

    Returns:
        Dict[int, List[str]]: Path index to list of service IDs.
    """
    mapping: Mapping[int, List[str]] = defaultdict(list)
    for svc in services:
        matched = set()
        for idx, path in paths_dict.items():
            for svc_path in infer_paths(svc):
                if shared_edges_between_services(svc_path, path):
                    mapping[idx].append(svc.id)
                    matched.add(svc.id)
                    break
    return mapping

def highlight_intersections(
    polygons: List[Polygon],
    ax: plt.Axes
) -> None:
    """
    Detect overlaps between polygons and highlight intersections, skipping non-area geometries.

    Args:
        polygons (List[Polygon]): List of polygons to check for intersections.
        ax (plt.Axes): Matplotlib axes to draw on.
    """
    for i, p1 in enumerate(polygons):
        for p2 in polygons[i + 1:]:
            inter = p1.intersection(p2)
            if inter.is_empty:
                continue
            # Only handle Polygon or MultiPolygon intersections
            if isinstance(inter, Polygon):
                parts = [inter]
            elif isinstance(inter, MultiPolygon):
                parts = list(inter)
            else:
                continue
            for part in parts:
                ax.add_patch(
                    MplPolygon(
                        list(part.exterior.coords), closed=True,
                        facecolor='crimson', edgecolor='crimson', alpha=0.5
                    )
                )

def add_markers_to_legend(markers: Mapping[str, Mapping[str, str]], ax: plt.Axes) -> None:
    """
    Add legend entries for markers used in the plot.

    Args:
        marker (Mapping[str, Mapping[str, str]): Dictionary of marker styles with marker as key and label as value.
        ax (plt.Axes): Matplotlib axes.
    """
    for _, marker_data in markers.items():
        marker = marker_data['marker']
        label = marker_data['label']
        ax.scatter([], [], marker=marker, s=100, edgecolors='black', linewidths=1.5, color='white', label=label)

def get_time_label(minutes: float, _pos) -> str:
    """
    Formatter for x-axis: minutes since midnight to HH:MM h.

    Args:
        minutes (float): Minutes from midnight.
        _pos: Required by FuncFormatter, unused.

    Returns:
        str: Formatted hours and minutes string.
    """
    hrs = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hrs:02d}:{mins:02d} h."

def configure_marey_axes(
    ax: plt.Axes,
    station_positions: Mapping[Station, float],
    min_x: int,
    max_x: int,
    title: str
) -> None:
    """
    Configure common axes properties for a Marey chart.

    Args:
        ax (plt.Axes): Axes to configure.
        station_positions (Mapping[Station, float]): Station y-positions.
        min_x (int): Minimum x-axis bound.
        max_x (int): Maximum x-axis bound.
        title (str): Title for the plot.
    """
    # Style spines
    for side in ('top', 'right', 'bottom', 'left'):
        spine = ax.spines[side]
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color('#A9A9A9')

    # Ticks and labels
    ax.tick_params(axis='both', which='major', labelsize=16)
    y_positions = list(station_positions.values())
    ax.set_yticks(y_positions)
    ax.set_yticklabels([station.name for station in station_positions.keys()], fontsize=16)

    # Grid
    ax.grid(True, color='#A9A9A9', alpha=0.3, linestyle='-', linewidth=1.0, zorder=1)

    # Axis limits
    x_range = max_x - min_x
    ax.set_xlim(-(min_x + 0.03 * x_range), max_x + 0.03 * x_range)

    # Title and axis labels
    ax.set_title(title, fontweight='bold', fontsize=24, pad=20)
    ax.set_xlabel('Time (HH:MM)', fontsize=18)
    ax.set_ylabel('Stations', fontsize=18)

    # X-axis formatting
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.xaxis.set_major_formatter(FuncFormatter(get_time_label))
    plt.setp(ax.get_xticklabels(), rotation=70, ha='right', fontsize=20)

def show_plot(fig: plt.Figure, save_path: str = None) -> None:
    """
    Show the plot and save it to a file if a path is provided.

    Args:
        fig (plt.Figure): Figure to show.
        save_path (str, optional): Path to save the plot in PDF format. Defaults to None.
    """
    plt.show()
    if save_path:
        save_path_dir = Path(save_path).parent
        save_path_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', transparent=True)

def build_service_schedule(
    services: List[Service],
    station_positions: Mapping[Station, float]
) -> Dict[str, Dict[Station, Tuple[int, int]]]:
    """
    Build arrival/departure minute offsets for each service at each station.

    Args:
        services (List[Service]): Services to process.
        station_positions (Mapping[Station, float]): Station positions.

    Returns:
        Dict[str, Dict[Station, Tuple[int, int]]]: service_id to {station: (arrival_min, departure_min)}.
    """
    schedule = {}
    for svc in services:
        base = svc.time_slot.start.total_seconds() // 60
        station_times = {}
        for station, (arrival, departure) in zip(svc.line.stations, svc.line.timetable.values()):
            if station in station_positions:
                arr = int(base + arrival)
                dep = int(base + departure)
                station_times[station] = (arr, dep)
        schedule[svc.id] = station_times
    return schedule

def prepare_service_colors(services: List[Service]) -> Dict[int, str]:
    """
    Prepare color mapping for services based on TSP names.

    Args:
        services (List[Service]): List of service objects.

    Returns:
        Dict[int, str]: Mapping of service IDs to hex color strings.
    """
    tsps = sorted({svc.tsp.name for svc in services})
    cmap = ListedColormap(sns.color_palette('pastel', len(tsps)).as_hex())
    tsp_color = {tsp: cmap(i) for i, tsp in enumerate(tsps)}
    return {svc.id: tsp_color[svc.tsp.name] for svc in services}

def update_time_bounds(
    schedule_times: Mapping[Station, Tuple[int, int]],
    min_x: int,
    max_x: int
) -> Tuple[int, int]:
    """
    Update the minimum and maximum time bounds based on schedule times.

    Args:
        schedule_times (Mapping[Station, List[int]]): Arrival/departure times per station.
        min_x (int): Current minimum x-axis bound.
        max_x (int): Current maximum x-axis bound.

    Returns:
        Tuple[int, int]: Updated (min_x, max_x) bounds.
    """
    times = [t for times in schedule_times.values() for t in times]
    return min(min_x, min(times)), max(max_x, max(times))

def plot_service_markers(
    ax: plt.Axes,
    service: Service,
    schedule_times: Mapping[Station, List[int]],
    station_positions: Mapping[Station, float],
    color: str,
    markers: Mapping[str, Mapping[str, str]]
) -> None:
    """
    Plot departure and arrival markers for a service.

    Args:
        ax (plt.Axes): Matplotlib axes.
        service (Service): Service to plot.
        schedule_times (Mapping[Station, List[int]]): Times per station.
        station_positions (Mapping[Station, float]): Station y-positions.
        color (str): Hex color for the service.
    """
    items = list(schedule_times.items())
    first_station, (arrival_first, departure_first) = items[0]
    last_station, (arrival_last, departure_last) = items[-1]
    is_first_station = first_station == service.line.stations[0]
    is_last_station = last_station == service.line.stations[-1]
    start_marker = markers['departure']['marker'] if is_first_station else markers['intermediate']['marker']
    end_marker = markers['arrival']['marker'] if is_last_station else markers['intermediate']['marker']

    ax.scatter(
        arrival_first,
        station_positions[first_station],
        marker=start_marker,
        s=100,
        edgecolors='black',
        linewidths=1.5,
        color=color,
        zorder=5,
    )
    ax.scatter(
        departure_last,
        station_positions[last_station],
        marker=end_marker,
        s=100,
        edgecolors='black',
        linewidths=1.5,
        color=color,
        zorder=5,
    )

def plot_service_line(
    ax: plt.Axes,
    service: Service,
    schedule_times: Mapping[Station, List[int]],
    station_positions: Mapping[Station, float],
    color: str
) -> None:
    """
    Plot the path line with markers for intermediate stations.

    Args:
        ax (plt.Axes): Matplotlib axes.
        service (Service): Service to plot.
        schedule_times (Mapping[Station, List[int]]): Times per station.
        station_positions (Mapping[Station, float]): Station y-positions.
        color (str): Hex color for the service.
    """
    points = [(time, station_positions[station]) for station, times in schedule_times.items() for time in times]
    ax.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        marker='o',
        linewidth=2.0,
        color=color,
        label=service.tsp.name if service.tsp.name not in ax.get_legend_handles_labels()[1] else None
    )

def make_safety_polygon(
    departure_time: int,
    arrival_time: int,
    y1: float,
    y2: float,
    gap: int
) -> Polygon:
    """
    Create a rectangular polygon around a segment for safety margin.

    Args:
        departure_time (int): departure minute.
        arrival_time (int): arrival minute.
        y1 (float): position of departure station.
        y2 (float): position of arrival station.
        gap (int): safety gap in minutes.

    Returns:
        Polygon: safety area polygon.
    """
    return Polygon([
        (departure_time - gap, y1),
        (arrival_time - gap, y2),
        (arrival_time + gap, y2),
        (departure_time + gap, y1)
    ])

def draw_safety_overlay(
    ax: plt.Axes,
    schedule_times: Mapping[Station, List[int]],
    station_positions: Mapping[Station, float],
    safety_gap: int
) -> List[Polygon]:
    """
    Create and add safety polygons between consecutive station stops.

    Args:
        ax (plt.Axes): Matplotlib axes to draw on.
        schedule_times (Mapping[Station, List[int]]): Times per station.
        station_positions (Mapping[Station, float]): Station y-positions.
        safety_gap (int): Minutes of buffer around each segment.

    Returns:
        List[Polygon]: List of safety polygons created.
    """
    polygons: List[Polygon] = []
    items = list(schedule_times.items())
    for (station_1, (_, departure_1)), (station_2, (arrival_2, _)) in zip(items, items[1:]):
        poly = make_safety_polygon(
            departure_1,
            arrival_2,
            station_positions[station_1],
            station_positions[station_2],
            safety_gap
        )
        polygons.append(poly)
        ax.add_patch(MplPolygon(
                list(poly.exterior.coords),
                closed=True,
                facecolor='#D3D3D3',
                edgecolor='#D3D3D3',
                alpha=0.6
            )
        )
    return polygons

def plot_path_marey(
    services: List[Service],
    station_positions: Mapping[Station, float],
    safety_gap: int,
    save_path: str,
    path_idx: int,
    markers: Mapping[str, Mapping[str, str]]
) -> None:
    """
    Render Marey chart for a single path.

    Args:
        services (List[Service]): Services assigned to this path.
        station_positions (Dict[Station, float]): Normalized station distances.
        safety_gap (int): Minutes of buffer around each segment.
        save_path (Optional[str]): Directory to save the plot.
        path_idx (int): Index of the path (for filename).
        markers (Mapping[str, Mapping[str, str]]): Markers for departure, arrival and intermediate stations.
    """
    service_colors = prepare_service_colors(services)
    fig, ax = plt.subplots(figsize=(20, 11))
    min_x, max_x = 0, 24 * 60
    schedule = build_service_schedule(services, station_positions)
    all_polygons: List[Polygon] = []
    for service in services:
        service_schedule = schedule[service.id]
        min_x, max_x = update_time_bounds(schedule_times=service_schedule, min_x=min_x, max_x=max_x)
        plot_service_markers(
            ax=ax,
            service=service,
            schedule_times=service_schedule,
            station_positions=station_positions,
            color=service_colors[service.id],
            markers=markers
        )
        plot_service_line(
            ax=ax,
            service=service,
            schedule_times=service_schedule,
            station_positions=station_positions,
            color=service_colors[service.id]
        )
        polygons = draw_safety_overlay(
            ax=ax,
            schedule_times=service_schedule,
            station_positions=station_positions,
            safety_gap=safety_gap
        )
        all_polygons.extend(polygons)
    highlight_intersections(all_polygons, ax)
    add_markers_to_legend(markers, ax)
    start_station = next(iter(station_positions.keys()))
    end_station = list(station_positions.keys())[-1]
    title = f"{start_station.name} - {end_station.name}"
    configure_marey_axes(ax, station_positions, min_x, max_x, title)
    ax.legend()
    plt.tight_layout()
    show_plot(fig, f'{save_path}{path_idx}.pdf')

def plot_marey_chart(
        supply: Supply,
        date: datetime.date,
        safety_gap: int = SAFETY_GAP,
        save_path: str = None,
        markers: Mapping[str, Tuple[str, str]] = MARKERS
) -> None:
    """
    Plot Marey chart for all corridors and paths on the given date.

    Args:
        supply (Supply): Supply object containing services and corridors.
        date (datetime.date): Date to filter services.
        safety_gap (int): Safety gap in minutes between segments.
        save_path (str, optional): Directory path to save PDF files. If None, charts are shown only.
        markers (Mapping[str, Tuple[str, str]], optional): Markers for departure and arrival.
    """
    services = supply.filter_services_by_date(date)
    corridors = set(supply.corridors)

    paths_dict = enumerate_unique_paths(corridors)
    paths_positions = compute_normalized_positions(paths_dict)
    services_paths = assign_services_to_paths(services, paths_dict)

    for path_idx, station_positions in paths_positions.items():
        service_ids = services_paths.get(path_idx, [])
        if not service_ids:
            continue

        services_in_path = [service for service in services if service.id in service_ids]
        plot_path_marey(
            services=services_in_path,
            station_positions=station_positions,
            safety_gap=safety_gap,
            save_path=save_path,
            path_idx=path_idx,
            markers=markers
        )
