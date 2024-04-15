"""Entities for Railway Market Tests."""

import cProfile as profile
import numpy as np
import pandas as pd
import time
import tqdm

from benchmarks.generator import get_revenue_behaviour
from benchmarks.robin_railway import RevenueMaximization
from benchmarks.utils import sns_line_plot, int_input, get_schedule_from_supply, TrainSchedulePlotter, infer_line_stations
from src.entities import GSA, Solution

from robin.kernel.entities import Kernel
from robin.plotter.entities import KernelPlotter
from robin.scraping.entities import SupplySaver
from robin.services_generator.entities import ServiceGenerator
from robin.supply.entities import Service, Supply

from pathlib import Path
from tqdm.notebook import tqdm
from typing import List, Union


class RailwayMarketDynamics:
    def __init__(self,
                 supply_config_path: Union[Path, None] = None,
                 demand_config_path: Union[Path, None] = None,
                 generator_config_path: Union[Path, None] = None,
                 generator_save_path: Union[Path, None] = None,
                 seed: Union[int, None] = None
                 ) -> None:

        if generator_config_path:
            n_services = int_input("Number of services to generate: ")
            generator = ServiceGenerator(supply_config_path=supply_config_path)
            _ = generator.generate(file_name=generator_save_path,
                                   path_config=generator_config_path,
                                   n_services=n_services,
                                   seed=seed)
            supply_config_path = generator_save_path
            print(f'Number of service requests generated: {len(_)}')

        self.supply_config_file = supply_config_path
        self.demand_config_path = demand_config_path
        self.seed = seed

    def run(self,
            gsa_supply_save_path: str,
            robin_save_path: Path,
            gsa_population: int = 20,
            gsa_iters: int = 50,
            gsa_runs: int = 10,
            gsa_chaotic: bool = True,
            gsa_verbosity: bool = False,
            ) -> List[Service]:
        global_train_hist = pd.DataFrame()
        runs_best_solution_history = {}
        supply = Supply.from_yaml(self.supply_config_file)
        requested_schedule = get_schedule_from_supply(self.supply_config_file)
        revenue_behaviour = get_revenue_behaviour(requested_schedule)
        lines = supply.lines
        line = infer_line_stations(lines)
        plotter = TrainSchedulePlotter(requested_schedule, line)
        print(requested_schedule)
        print(line)
        plotter.plot(save_path=Path('../figures/requested_schedule.pdf'))
        plotter.plot_security_gaps()

        for r in tqdm(range(1, gsa_runs + 1)):
            sm = RevenueMaximization(requested_schedule=requested_schedule,
                                     revenue_behaviour=revenue_behaviour,
                                     line=line,
                                     safe_headway=10)

            gsa_algo = GSA(objective_function=sm.get_fitness_gsa,
                           is_feasible=sm.feasible_services_times,
                           custom_repair=sm.custom_repair,
                           r_dim=len(sm.boundaries.real),
                           d_dim=0,
                           boundaries=sm.boundaries)

            pr = profile.Profile()
            pr.disable()

            pr.enable()
            training_history = gsa_algo.optimize(population_size=gsa_population,
                                                 iters=gsa_iters,
                                                 chaotic_constant=gsa_chaotic,
                                                 repair_solution=True,
                                                 initial_population=sm.get_initial_population(gsa_population),
                                                 verbose=gsa_verbosity)
            pr.disable()
            pr.dump_stats('profile.pstat')

            training_history.insert(0, "Run", r)
            training_history['Discrete'] = [sm.best_solution.discrete for _ in range(len(training_history))]
            global_train_hist = pd.concat([global_train_hist, training_history], axis=0)

            runs_best_solution_history[r] = (sm.best_solution, sm.best_revenue)

        # Table with results by run
        dtypes = {'Run': np.int_,
                  'Revenue': np.float64,
                  'Execution Time (s.)': np.float64,
                  'Scheduled Trains': np.int_,
                  'Delta DT (min.)': np.float64,
                  'Delta TT (min.)': np.float64}

        summary_df = pd.DataFrame(columns=list(dtypes.keys()))

        run_grouped_df = global_train_hist.groupby('Run')
        for group in run_grouped_df.groups:
            run = run_grouped_df.get_group(group)['Run'].iloc[-1]
            revenue = run_grouped_df.get_group(group)['Fitness'].iloc[-1]
            execution_time = run_grouped_df.get_group(group)['ExecutionTime'].iloc[-1]
            scheduled_trains_array = run_grouped_df.get_group(group)['Discrete'].iloc[-1]
            scheduled_trains = int(sum(run_grouped_df.get_group(group)['Discrete'].iloc[-1]))
            real_solution = run_grouped_df.get_group(group)['Real'].iloc[-1]
            sm.update_schedule(Solution(real=real_solution, discrete=scheduled_trains))
            delta_dt = 0.0
            delta_tt = 0.0
            for i, service in enumerate(sm.requested_schedule):
                if not scheduled_trains_array[i]:
                    continue
                departure_station = list(sm.requested_schedule[service].keys())[0]
                delta_dt += abs(sm.updated_schedule[service][departure_station][1] -
                                sm.requested_schedule[service][departure_station][1])
                for j, stop in enumerate(sm.requested_schedule[service].keys()):
                    if j == 0 or j == len(sm.requested_schedule[service]) - 1:
                        continue
                    delta_tt += abs(sm.updated_schedule[service][stop][1] - sm.requested_schedule[service][stop][1])

            summary_df.loc[len(summary_df)] = [run, revenue, execution_time, scheduled_trains, delta_dt, delta_tt]

        summary_df = summary_df.sort_values('Revenue', ascending=False)
        display(summary_df)

        for col in dtypes:
            summary_df[col] = summary_df[col].astype(dtypes[col])

        # Global status
        print("Global GSA status:")

        # Execution time (mean and std.)
        run_times = run_grouped_df['ExecutionTime'].last()
        print(f'\tTotal execution time: {round(run_times.sum(), 4)} s.')
        print(f'\tExecution Time (by run) - Mean: {round(run_times.mean(), 4)} s. - Std: {round(run_times.std(), 4)} s.')

        # Revenue (mean and std.)
        run_revenues = run_grouped_df['Fitness'].last()
        print(f'\tRevenue - Mean: {round(run_revenues.mean(), 4)} - Std: {round(run_revenues.std(), 4)}')

        # Scheduled trains (mean and std.)
        run_trains = run_grouped_df['Discrete'].last().apply(sum)
        print(f'\tScheduled Trains - Mean: {np.round(run_trains.mean())} - Std: {np.round(run_trains.std())}')

        runs_best_solution_history = dict(sorted(runs_best_solution_history.items(),
                                                 key=lambda x: x[1][1]))

        q2_solution_index = np.floor(gsa_runs // 2).astype(int)
        gsa_solution = tuple(runs_best_solution_history.items())[q2_solution_index]

        print(f"\tMedian solution: Run {gsa_solution[0]}")
        max_revenue = sum([sm.revenue[service]['canon'] for service in sm.revenue])
        print(f"\tMax Revenue: {max_revenue} - WARNING!: Scheduling all services could not be feasible")

        # GSA Convergence plot
        sns_line_plot(df=global_train_hist,
                      x_data="Iteration",
                      y_data="Fitness",
                      title="GSA Convergence",
                      x_label="Iteration",
                      y_label="Fitness (Revenue)")

        services = sm.update_supply(path=self.supply_config_file,
                                    solution=gsa_solution[1][0])

        filtered_services = {}
        for i, service in enumerate(sm.updated_schedule):
            if gsa_solution[1][0].discrete[i]:
                filtered_services[service] = sm.updated_schedule[service]

        print(filtered_services)
        plotter = TrainSchedulePlotter(filtered_services, line)
        plotter.plot(save_path=Path('../figures/updated.pdf'))
        plotter.plot_security_gaps()

        tt_file_name = f'{self.supply_config_file.stem}_gsa'
        SupplySaver(services).to_yaml(filename=f'{tt_file_name}.yml', save_path=gsa_supply_save_path)
        self.supply_config_file = Path(f'{gsa_supply_save_path}{tt_file_name}.yml')

        # Simulate market
        kernel = Kernel(path_config_supply=self.supply_config_file,
                        path_config_demand=self.demand_config_path)

        services = kernel.simulate(output_path=robin_save_path, departure_time_hard_restriction=True)

        kernel_plotter = KernelPlotter(path_output_csv=robin_save_path,
                                       path_config_supply=self.supply_config_file)

        kernel_plotter.plotter_data_analysis()
        kernel_plotter.plot_users_seat_pie_chart()

        return services
