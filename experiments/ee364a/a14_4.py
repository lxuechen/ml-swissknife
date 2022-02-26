import cvxpy as cp
import fire
import numpy as np

n = 10  # number of variables
k = 6  # number of designs

# component widths from known designas
# each column of W is a different design
W = ([[1.8381, 1.5803, 12.4483, 4.4542, 6.5637, 5.8225],
      [1.0196, 3.0467, 18.4965, 3.6186, 7.6979, 2.3292],
      [1.6813, 1.9083, 17.3244, 4.677, 4.6581, 27.0291],
      [1.3795, 2.625, 14.6737, 4.1361, 7.161, 7.5759],
      [1.8318, 1.4526, 17.2696, 3.7408, 2.2107, 10.3642],
      [1.5028, 3.0937, 14.9034, 4.4055, 7.8582, 20.5204],
      [1.7095, 2.1351, 10.1296, 4.0931, 2.9001, 9.9634],
      [1.4289, 3.58, 9.3459, 3.8898, 2.7663, 15.1383],
      [1.3046, 3.561, 10.1179, 4.3891, 7.1302, 3.8139],
      [1.1897, 2.7807, 13.0112, 4.2426, 6.1611, 29.6734]])
W = np.array(W)  # n x k

(W_min, W_max) = (1.0, 30.0)

# objective values for the different designs
# entry j gives the objective for design j
P = np.array([29.0148, 46.3369, 282.1749, 78.5183, 104.8087, 253.5439])
D = np.array([15.9522, 11.5012, 4.8148, 8.5697, 8.087, 6.0273])
A = np.array([22.3796, 38.7908, 204.1574, 62.5563, 81.2272, 200.5119])

# specifications
(P_spec, D_spec, A_spec) = (60.0, 10.0, 50.0)


def main():
    logP, logD, logA = [np.log(arr) for arr in (P, D, A)]
    theta = cp.Variable(shape=(k,))
    objective = cp.Minimize(0)
    logtarget = cp.matmul(np.log(W), theta)
    target = cp.exp(logtarget)
    constraints = [
        sum(theta) == 1,
        theta >= 0,
        theta @ logP <= np.log(P_spec),
        theta @ logD <= np.log(D_spec),
        theta @ logA <= np.log(A_spec),
        np.log(W_min) <= logtarget,
        logtarget <= np.log(W_max),
    ]
    cp.Problem(objective, constraints).solve()
    print(theta.value)
    print(target.value)


if __name__ == "__main__":
    fire.Fire(main)
