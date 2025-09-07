import numpy as np

def sphere(x):
    return np.sum(x**2)

def schwefel_2_22(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def schwefel_1_2(x):
    return np.sum([np.sum(x[:i+1])**2 for i in range(len(x))])

def schwefel_2_21(x):
    return np.max(np.abs(x))

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

def step(x):
    return np.sum((x + 0.5)**2)

def quartic_noise(x):
    return np.sum(np.arange(1, len(x)+1) * x**4) + np.random.uniform(0, 1)

def schwefel(x):
    return -np.sum(x * np.sin(np.sqrt(np.abs(x))))

def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def ackley(x):
    n = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2)/n)) \
           - np.exp(np.sum(np.cos(2 * np.pi * x))/n) + 20 + np.e

def griewank(x):
    return np.sum(x**2)/4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1)))) + 1

def penalized_1(x):
    a, k, m = 10, 100, 4
    y = 1 + (x + 1) / 4
    term1 = np.pi / len(x) * (
        10 * np.sin(np.pi * y[0])**2
        + np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:])**2))
        + (y[-1] - 1)**2
    )
    def u_fun(xi):
        if xi > a:
            return k * (xi - a)**m
        elif xi < -a:
            return k * (-xi - a)**m
        else:
            return 0
    term2 = np.sum([u_fun(xi) for xi in x])
    return term1 + term2


class func:
    def __init__(self, func, lower_bound, upper_bound):
        self.func  = func
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound


functions = {
    "Sphere":            func(sphere,        -100,   100),
    "Schwefel 2.22":     func(schwefel_2_22,  -10,    10),
    "Schwefel 1.2":      func(schwefel_1_2,  -100,   100),
    "Schwefel 2.21":     func(schwefel_2_21, -100,   100),
    "Rosenbrock":        func(rosenbrock,    -30,    30),
    "Step":              func(step,         -100,   100),
    "Quartic with Noise":func(quartic_noise,-128,   128),
    "Schwefel":          func(schwefel,     -500,   500),
    "Rastrigin":         func(rastrigin,    -5.12,  5.12),
    "Ackley":            func(ackley,        -32,    32),
    "Griewank":          func(griewank,    -600,   600),
    "Penalized 1":       func(penalized_1,  -50,    50),
}

formulas = {
    "Sphere":            r"f(x) = \sum_{i=1}^{n} x_i^2",
    "Schwefel 2.22":     r"f(x) = \sum |x_i| + \prod |x_i|",
    "Schwefel 1.2":      r"f(x) = \sum_{i=1}^n \Bigl(\sum_{j=1}^i x_j\Bigr)^2",
    "Schwefel 2.21":     r"f(x) = \max_i |x_i|",
    "Rosenbrock":        r"f(x) = \sum_{i=1}^{n-1}\!\bigl[100(x_{i+1}-x_i^2)^2+(x_i-1)^2\bigr]",
    "Step":              r"f(x) = \sum_{i=1}^{n}(x_i + 0.5)^2",
    "Quartic with Noise":r"f(x) = \sum_{i=1}^{n}i\,x_i^4 + \text{random}[0,1]",
    "Schwefel":          r"f(x) = -\sum_i x_i\sin(\sqrt{|x_i|})",
    "Rastrigin":         r"f(x) = 10n + \sum_i\bigl(x_i^2 - 10\cos(2\pi x_i)\bigr)",
    "Ackley":            r"f(x) = -20e^{-0.2\sqrt{\frac1n\sum x_i^2}} - e^{\frac1n\sum\cos2\pi x_i}+20+e",
    "Griewank":          r"f(x)=\frac1{4000}\sum x_i^2 - \prod_i\cos\bigl(x_i/\sqrt{i}\bigr)+1",
    "Penalized 1":       r"f(x)=\frac\pi n\bigl[10\sin^2\pi y_1+\sum(y_i-1)^2(1+10\sin^2\pi y_{i+1})+(y_n-1)^2\bigr]+\sum u(x_i)",
}

__all__ = ["functions", "formulas"]