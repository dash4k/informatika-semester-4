import matplotlib.animation as animation
import os
from gwo import GWO, Logger
from utils import Spinner
import matplotlib.pyplot as plt
import numpy as np
from functions import functions
from sklearn.decomposition import PCA
 
dimension_list = [10, 20, 30, 40]

for fname, f in functions.items():
    print(f"\n--- Tuning for function: {fname} ---")

    best_fitness = float("inf")
    best_dimension_val = None

    for dimension in dimension_list:
        result_dir = f"results/{fname}/{dimension}_dimension"
        os.makedirs(result_dir, exist_ok=True)

        logger = Logger(f"{result_dir}/log.txt")
        
        gwo = GWO(func=f.func, search_agents=30, max_iter=100, dim=dimension, lb=f.lower_bound, ub=f.upper_bound)
        spinner = Spinner(f"{fname} ({dimension} dimension)")
        fitness, _, curve, position_history = gwo.fit(logger=logger, spinner=spinner)

        plot_path = f"{result_dir}/plot.png"
        plt.figure()
        plt.plot(curve)
        plt.title(f"{fname} Convergence Curve ({dimension}D)")
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        if position_history:
            all_positions = np.concatenate(position_history, axis=0)
            pca = PCA(n_components=2)
            pca.fit(all_positions)

            projected_history = [pca.transform(pos) for pos in position_history]

            fig, ax = plt.subplots()
            scat = ax.scatter([], [], c='blue', s=40)

            def update(frame):
                pos_2d = projected_history[frame]
                scat.set_offsets(pos_2d)
                ax.set_title(f"{fname} - Iteration {frame}")
                return scat,

            all_proj = np.vstack(projected_history)
            ax.set_xlim(all_proj[:, 0].min(), all_proj[:, 0].max())
            ax.set_ylim(all_proj[:, 1].min(), all_proj[:, 1].max())

            ani = animation.FuncAnimation(
                fig, update, frames=len(projected_history), interval=200, blit=True
            )
            ani.save(f"{result_dir}/wolf_positions.gif", writer='pillow')
            plt.close()

        if fitness < best_fitness:
            best_fitness = fitness
            best_dimension_val = dimension

    logger = Logger(f"results/{fname}/best.txt")
    logger.log(f"Best config for {fname}: dimension={best_dimension_val}")
    logger.log(f"Best fitness for {fname}: {best_fitness}")