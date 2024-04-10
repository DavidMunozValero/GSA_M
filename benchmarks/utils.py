"""Utils for benchmarks."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from typing import Mapping

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
