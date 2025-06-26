from gwo import GWO, Logger
from utils import Spinner, func
import matplotlib.pyplot as plt
import numpy as np
from functions import functions
 
dimention_list = [10, 20, 30, 40]

for fname, f in functions.items():
    print(f"\n--- Tuning for function: {fname} ---")

    best_config = None
    best_fitness = float("inf")

    for dimentions in dimention_list:
        logger = Logger(f"results/{fname}/{dimentions}_dimention/log.txt")

        search_agents = max(20, min(100, dimentions * 2))
        
        gwo = GWO(func=f.func, search_agents=search_agents, max_iter=100, dim=dimentions, lb=f.lower_bound, ub=f.upper_bound)
        spinner = Spinner(f"{fname} ({dimentions} dimention)")
        fitness, _, curve = gwo.fit(logger=logger, spinner=spinner)
        
        plot_path = f"results/{fname}/{dimentions}_dimention/plot.png"
        plt.figure()
        plt.plot(curve)
        plt.title(f"{fname} Convergence Curve ({dimentions} dimention)")
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        if fitness < best_fitness:
            if fitness < 1e-6:
                fitness = 0.0
            best_fitness = fitness
            best_dimentions_val = dimentions

    logger = Logger(f"results/{fname}/best.txt")
    logger.log(f"Best config for {fname}: dimentions={best_dimentions_val}")
    logger.log(f"Best fitness for {fname}: {best_fitness}")