import numpy as np
from util import Logger

def initialize_particles(n_particles, dim):
    particles = np.array([
        [1.0, -2.0],
        [-3.0, 1.5],
        [0.5, 0.5],
        [4.0, -1.0],
        [-1.0, 3.0]
    ])
    velocities = np.zeros((n_particles, dim))
    return particles, velocities

def update_velocity_position(particles, velocities, personal_best, g_best, w, c1, c2, bound, t):
    r_values = {
        0: (0.2, 0.2),
        1: (0.2, 0.5),
        2: (0.2, 0.8),
        3: (0.8, 0.2),
        4: (0.2, 0.8)
    }

    r1_scalar, r2_scalar = r_values[t]
    n_particles, dim = particles.shape
    r1 = np.full((n_particles, dim), r1_scalar)
    r2 = np.full((n_particles, dim), r2_scalar)

    velocities = (
        w * velocities
        + c1 * r1 * (personal_best - particles)
        + c2 * r2 * (g_best - particles)
    )
    particles = particles + velocities
    particles = np.clip(particles, bound[0], bound[1])

    return particles, velocities

def evaluate_particles(particles, func, personal_best, personal_best_val, g_best, g_best_val):
    for i in range(len(particles)):
        f_val = func(particles[i])
        if f_val < personal_best_val[i]:
            personal_best[i] = particles[i].copy()
            personal_best_val[i] = f_val
            if f_val < g_best_val:
                g_best = particles[i].copy()
                g_best_val = f_val
    return personal_best, personal_best_val, g_best, g_best_val


class PSO:
    def __init__(self, func, dim, lb, ub):
        self.func = func
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.n_particles = 5
        self.max_iter = 5

        self.w = 0.7
        self.c1 = 2.0
        self.c2 = 2.0

        self.positions, self.velocities = initialize_particles(self.n_particles, self.dim)

        self.personal_best = self.positions.copy()
        self.personal_best_val = np.array([float('inf')] * self.n_particles)

        self.global_best = np.zeros(self.dim)
        self.global_best_val = float('inf')

    def fit(self, logger=None):
        convergence_curve = []
        position_history = []

        for t in range(self.max_iter):
            self.positions, self.velocities = update_velocity_position(
                self.positions, self.velocities,
                self.personal_best, self.global_best,
                self.w, self.c1, self.c2,
                (self.lb, self.ub), t
            )

            self.personal_best, self.personal_best_val, self.global_best, self.global_best_val = evaluate_particles(
                self.positions, self.func,
                self.personal_best, self.personal_best_val,
                self.global_best, self.global_best_val
            )

            convergence_curve.append(self.global_best_val)
            position_history.append(self.positions.copy())

            if logger:
                logger.log(f"Iteration {t+1}, Best Fitness: {self.global_best_val:.6f}")

        if logger:
            logger.log(f"\nFinal Best Fitness: {self.global_best_val}")
            logger.log(f"Best Position Found:\n{self.global_best}")

        return self.global_best_val, self.global_best, convergence_curve, position_history
    

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    def sphere(x):
        return sum(xi ** 2 for xi in x)

    result_dir = "results/PSO"
    os.makedirs(result_dir, exist_ok=True)

    logger = Logger(f"{result_dir}/log.txt")

    model = PSO(func=sphere, dim=2, lb=-5.12, ub=5.12)
    fitness, best_pos, curve, position_history = model.fit(logger=logger)

    plot_path = f"{result_dir}/plot.png"
    plt.figure()
    plt.plot(curve)
    plt.title("PSO Convergence Curve (2D)")
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
        scat = ax.scatter([], [], c='green', s=40)

        def update(frame):
            pos_2d = projected_history[frame]
            scat.set_offsets(pos_2d)
            ax.set_title(f"PSO - Iteration {frame+1}")
            return scat,

        all_proj = np.vstack(projected_history)
        ax.set_xlim(all_proj[:, 0].min(), all_proj[:, 0].max())
        ax.set_ylim(all_proj[:, 1].min(), all_proj[:, 1].max())

        import matplotlib.animation as animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(projected_history), interval=1000, blit=True
        )
        ani.save(f"{result_dir}/particle_positions.gif", writer='pillow')
        plt.close()

    logger.log("Best config for Sphere with PSO: dimension=2")
    logger.log(f"Best fitness for Sphere: {fitness}")
    logger.log(f"Best position: {best_pos}")