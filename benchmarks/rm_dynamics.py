"""Entities for Railway Market Tests."""

import pandas as pd
import tqdm

from benchmarks.robin_railway import RevenueMaximization
from benchmarks.utils import sns_line_plot
from src.entities import GSA

from robin.kernel.entities import Kernel
from robin.scraping.entities import SupplySaver
from robin.services_generator.entities import ServiceGenerator
from robin.supply.entities import Service

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
            generator = ServiceGenerator(supply_config_path=supply_config_path)
            generator.generate(file_name=generator_save_path,
                               path_config=generator_config_path,
                               n_services=5,
                               seed=seed)
            supply_config_path = generator_save_path

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
        for r in tqdm(range(1, gsa_runs + 1)):
            print(self.supply_config_file)
            sm = RevenueMaximization(supply_config_path=self.supply_config_file, safe_headway=10)

            gsa_algo = GSA(objective_function=sm.get_fitness_gsa,
                           is_feasible=sm.feasible_services_times,
                           custom_repair=sm.custom_repair,
                           r_dim=len(sm.boundaries.real),
                           d_dim=0,
                           boundaries=sm.boundaries)

            training_history = gsa_algo.optimize(population_size=gsa_population,
                                                 iters=gsa_iters,
                                                 chaotic_constant=gsa_chaotic,
                                                 repair_solution=True,
                                                 initial_population=sm.get_initial_population(gsa_population),
                                                 verbose=gsa_verbosity)
            training_history.insert(0, "run", r)
            global_train_hist = pd.concat([global_train_hist, training_history], axis=0)

            runs_best_solution_history[sm.best_revenue] = sm.best_solution

        # GSA Convergence plot
        sns_line_plot(df=global_train_hist,
                      x_data="Iteration",
                      y_data="Fitness",
                      title="GSA Convergence",
                      x_label="Iteration",
                      y_label="Fitness (Revenue)")

        # Execution time (mean and std.)
        run_times = global_train_hist.groupby('run')['ExecutionTime'].last()
        print(f'Mean: {round(run_times.mean(), 4)}s. - Std: {round(run_times.std(), 4)}s.')

        runs_best_solution_history = dict(sorted(runs_best_solution_history.items(), key=lambda x: x[0], reverse=True))
        print(runs_best_solution_history)
        gsa_solution = tuple(runs_best_solution_history.values())[min(0, gsa_runs // 2 - 1)]

        services = sm.update_supply(path=self.supply_config_file,
                                    solution=gsa_solution)

        tt_file_name = f'{self.supply_config_file.stem}_gsa'
        SupplySaver(services).to_yaml(filename=f'{tt_file_name}.yml', save_path=gsa_supply_save_path)
        self.supply_config_file = Path(f'{gsa_supply_save_path}{tt_file_name}.yml')

        # Simulate market
        kernel = Kernel(path_config_supply=self.supply_config_file,
                        path_config_demand=self.demand_config_path)

        services = kernel.simulate(output_path=robin_save_path, departure_time_hard_restriction=True)
        return services
