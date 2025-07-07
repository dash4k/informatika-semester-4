import os
import numpy as np
from util import Logger


class GWO:
    def __init__(self, func, dim, lb, ub, search_agents=5, max_iter=5):
        self.func = func
        self.search_agents = search_agents  #Number of wolves
        self.max_iter = max_iter
        self.dim = dim
        self.lb = lb
        self.ub = ub

        self.positions = np.array([
            [1.0, -2.0],
            [-3.0, 1.5],
            [0.5, 0.5],
            [4.0, -1.0],
            [-1.0, 3.0]
        ])

        self.r_values = [
            (0.1, 0.4),
            (0.6, 0.4),
            (0.1, 0.4),
            (0.6, 0.4),
            (0.1, 0.4)
        ]

    def fit(self, logger=None):
        alpha, beta, delta = np.zeros(self.dim), np.zeros(self.dim), np.zeros(self.dim)
        alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')
        convergence_curve = []
        position_history = []

        for t in range(self.max_iter):
            for i in range(self.search_agents):
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                fitness = self.func(self.positions[i])
                if fitness < alpha_score:
                    alpha_score = fitness
                    alpha = self.positions[i].copy()
                elif fitness < beta_score:
                    beta_score = fitness
                    beta = self.positions[i].copy()
                elif fitness < delta_score:
                    delta_score = fitness
                    delta = self.positions[i].copy()

            a = 2 - t * (2 / self.max_iter)
            r1, r2 = self.r_values[t]

            for i in range(self.search_agents):
                for j in range(self.dim):
                    a1 = 2 * a * r1 - a
                    c1 = 2 * r2
                    d_alpha = abs(c1 * alpha[j] - self.positions[i][j])
                    x1 = alpha[j] - a1 * d_alpha

                    x2 = beta[j] - a1 * abs(c1 * beta[j] - self.positions[i][j])
                    x3 = delta[j] - a1 * abs(c1 * delta[j] - self.positions[i][j])

                    self.positions[i][j] = (x1 + x2 + x3) / 3

            convergence_curve.append(alpha_score)
            position_history.append(self.positions.copy())

            if logger:
                logger.log(f"Iteration {t+1}, Best Fitness: {alpha_score:.6f}")

        if logger:
            logger.log(f"\nFinal Best Fitness: {alpha_score}")
            logger.log(f"Best Position Found:\n{alpha}")

        return alpha_score, alpha, convergence_curve, position_history
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from gwo import GWO, Logger
    from sklearn.decomposition import PCA


    def sphere(x):
        return sum(xi ** 2 for xi in x)

    result_dir = "results/GWO"
    os.makedirs(result_dir, exist_ok=True)

    logger = Logger(f"{result_dir}/log.txt")

    model = GWO(func=sphere, dim=2, lb=-5.12, ub=5.12)
    fitness, best_pos, curve, position_history = model.fit(logger=logger)

    plot_path = f"{result_dir}/plot.png"
    plt.figure()
    plt.plot(curve)
    plt.title("Sphere Convergence Curve (2D)")
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
            ax.set_title(f"Sphere - Iteration {frame+1}")
            return scat,

        all_proj = np.vstack(projected_history)
        ax.set_xlim(all_proj[:, 0].min(), all_proj[:, 0].max())
        ax.set_ylim(all_proj[:, 1].min(), all_proj[:, 1].max())

        import matplotlib.animation as animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(projected_history), interval=1000, blit=True
        )
        ani.save(f"{result_dir}/wolf_positions.gif", writer='pillow')
        plt.close()

    logger.log("Best config for Sphere: dimension=2")
    logger.log(f"Best fitness for Sphere: {fitness}")
    logger.log(f"Best position: {best_pos}")