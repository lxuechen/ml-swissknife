import fire
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm, laplace, uniform

# True weights for the underlying mixture
true_mixture_weights = [.3, .5, .2]
N = 100
k = 3


# This function generates N from the true mixture of distributions.
# When you use this function to plot your estimated densities,
# pass in the weights you obtain instead of the default parameters.
def generate_samples(mixture_weights):
    np.random.seed(1)
    # Parameters for the distributions in our mixture: N(3, 2^2), U(-1, 2), L(-2, 3)
    samples = np.vstack([[np.random.normal(3, 2),
                          np.random.uniform(-1, 2),
                          np.random.laplace(-2, 3)] for _ in range(N)])
    indices = np.random.choice(list(range(len(mixture_weights))), p=mixture_weights, size=N)
    samples = np.take_along_axis(samples, indices[:, None], axis=1)
    return samples.flatten()


# This function evaluates a set of univariate samples
# under the Gaussian, Uniform, and Laplace densities in the problem.
def evaluate_density(samples):
    # Parameters for the distributions in our mixture: N(3, 2^2), U(-1, 2), L(-2, 3)
    # Minor note: Scipy uniform.pdf expects the first argument as the start interval and the
    # second as the length of interval, *not* the endpoint
    densities = [[norm.pdf(x, 3, 2),
                  uniform.pdf(x, -1, 3),
                  laplace.pdf(x, -2, 3)] for x in samples]
    densities = np.array(densities)
    return densities


# This function takes as input the estimated mixture weights
# and generates a plot of both the estimated and true density
# by analytically evaluating the density at each point in a discretized
# interval [-20, 20]
def plot_estimated_and_true_density(estimated_mixture_weights):
    discretized_x = np.linspace(-20, 20, num=100)
    all_densities = evaluate_density(discretized_x)
    estimated_densities = all_densities @ estimated_mixture_weights
    true_densities = all_densities @ true_mixture_weights
    plt.plot(discretized_x, estimated_densities, label="Estimated")
    plt.plot(discretized_x, true_densities, label="True")
    plt.title("True vs Estimated Density")
    plt.legend()
    plt.show()


samples = generate_samples(true_mixture_weights)
densities = evaluate_density(samples)

import cvxpy as cp


def main():
    lamb = cp.Variable(k)
    con = [lamb >= 0., cp.sum(lamb) == 1.]
    obj = cp.Maximize(sum(cp.log(lamb @ row) for row in densities))
    cp.Problem(constraints=con, objective=obj).solve()

    plot_estimated_and_true_density(lamb.value)
    print(lamb.value)


if __name__ == "__main__":
    fire.Fire(main)
