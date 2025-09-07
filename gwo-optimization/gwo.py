import os
import numpy as np

class Logger:
    def __init__(self, filename="logs/optimization_log.txt"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.filename = filename

        with open(self.filename, 'w') as f:
            f.write("Optimization Log\n")

    def log(self, message):
        with open(self.filename, 'a') as f:
            f.write(message + "\n")

class GWO:
    def __init__(self, func, search_agents, max_iter, dim, lb, ub):
        self.func = func
        self.search_agents = search_agents
        self.max_iter = max_iter
        self.dim = dim
        self.lb = lb
        self.ub = ub

    def fit(self, logger=None, spinner=None):
        positions = np.random.uniform(self.lb, self.ub, (self.search_agents, self.dim))

        alpha, beta, delta = np.zeros(self.dim), np.zeros(self.dim), np.zeros(self.dim)
        alpha_score, beta_score, delta_score = float('inf'), float('inf'), float('inf')
        convergence_curve = []
        position_history = []

        for t in range(self.max_iter):
            for i in range(self.search_agents):
                positions[i] = np.clip(positions[i], self.lb, self.ub)
                fitness = self.func(positions[i])
                if fitness < alpha_score:
                    alpha_score = fitness
                    alpha = positions[i].copy()
                
                elif fitness < beta_score:
                    beta_score = fitness
                    beta = positions[i].copy()
                
                elif fitness < delta_score:
                    delta_score = fitness
                    delta = positions[i].copy()

            a = 2 - t * (2 / self.max_iter)

            for i in range(self.search_agents):
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    a1 = 2 * a * r1 - a
                    c1 = 2 * r2
                    d_alpha = abs(c1 * alpha[j] - positions[i][j])
                    x1 = alpha[j] - a1 * d_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    a2 = 2 * a * r1 - a
                    c2 = 2 * r2
                    d_beta = abs(c2 * beta[j] - positions[i][j])
                    x2 = beta[j] - a2 * d_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    a3 = 2 * a * r1 - a
                    c3 = 2 * r2
                    d_delta = abs(c3 * delta[j] - positions[i][j])
                    x3 = delta[j] - a3 * d_delta

                    positions[i][j] = (x1 + x2 + x3) / 3

            convergence_curve.append(alpha_score)
            position_history.append(positions.copy())


            if logger:
                logger.log(f"Iteration {t}, Best Fitness: {alpha_score:.6f}")
            if spinner:
                spinner.spin()

        if logger:
            logger.log(f"\nFinal Best Fitness: {alpha_score}")
            logger.log(f"Best Position Found:\n{alpha}")
        
        print()
        return alpha_score, alpha, convergence_curve, position_history