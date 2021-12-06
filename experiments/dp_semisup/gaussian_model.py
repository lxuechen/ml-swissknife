"""
Two questions:
    1. (fairness) disparate impact in terms of convergence rate? epsilon, d, n dependence?
    2. (self-training) does self-training post-processing boost performance?

Where does randomness come from?
    a. sampled data (both labeled and unlabeled)
    b. dp noise
    c. test data
"""

import math

import fire
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from swissknife import utils


def make_labeled_data(n, d, sigma, prob, mu, group_id=2):
    # mu: tensor of size (1, d,)
    if group_id == 2:  # Sample both groups.
        coin_flips = torch.bernoulli(torch.full(size=(n, 1), fill_value=prob))
        y = 2 * coin_flips - 1.  # Convert to Rademacher.
    elif group_id == 1:  # y=1 (bernoulli=1).
        y = torch.full(size=(n, 1), fill_value=1.)
    elif group_id == 0:  # y=-1 (bernoulli=0).
        y = torch.full(size=(n, 1), fill_value=-1.)
    else:
        raise ValueError(f"Unknown group_id: {group_id}")
    x = mu * y + sigma * torch.randn(size=(n, d))  # Get features.
    assert x.size() == (n, d) and y.size() == (n, 1)
    return x, y


def make_unlabeled_data(n, d, sigma, mu, prob, group_id=2):
    x, _ = make_labeled_data(n=n, d=d, sigma=sigma, mu=mu, prob=prob, group_id=group_id)  # Throw away labels.
    return x


def dp_estimator(x, y, clipping_norm, epsilon, delta=None):
    # Clip features.
    coefficient = torch.clamp_max(clipping_norm / (x.norm(dim=1, keepdim=True) + 1e-7), 1.)
    x = x * coefficient
    estimator = (x * y).mean(dim=0, keepdim=True)

    # Usual Gaussian mechanism.
    sample_size = x.size(0)
    if delta is None:
        delta = 1 / (sample_size ** 1.1)
    sensitivity = 2 / sample_size * clipping_norm
    var = 2 * math.log(1.25 / delta) * sensitivity ** 2 / (epsilon ** 2)

    return estimator + torch.randn_like(estimator) * math.sqrt(var)
    # -- correct


def usual_estimator(x, y, clipping_norm=None):
    if clipping_norm is not None:
        coefficient = torch.clamp_max(clipping_norm / x.norm(dim=1, keepdim=True), 1.)
        x = x * coefficient
    return (x * y).mean(dim=0, keepdim=True)


def classify(estimator, x):
    y_hat = torch.where((estimator * x).sum(dim=1, keepdim=True) > 0, 1., -1.)
    return y_hat


def compute_accuracy(estimator, x, y, _print):
    y_hat = classify(estimator=estimator, x=x)
    if _print:
        print(torch.eq(y, y_hat).float().mean(dim=0).item())
    return (1. - (y - y_hat).abs().mean() / 2).item()


def fairness(**kwargs):
    """Compare majority vs minority group's error.

    Plot error as a function of d, n, p (imbalance ratio), epsilon.

    Plot1: Check rate dependence on epsilon (same plot multiple p).
    Plot2: Check rate dependence on d.
    Plot3: Check rate dependence on p (same plot multiple epsilon).

    Maybe also plot ratio between accuracies? Need to reduce variance!
    Expectation:
        - O(1/p) vs O(1/(1-p)); not very surprising.
        - example too simple; doing well on majority => doing well on minority.

    Conclusion: This model is unsuited for fairness.
    """
    epsilons = (0.001, 0.01, 0.1, 0.5, 1., 2., 5.,)
    probs = (0.999, 0.9999)
    d = 20
    mu = torch.randn((1, d)) * 2
    sigma = 0.3
    n_labeled = 100
    n_test = 10000
    clipping_norm = 3
    seeds = list(range(100))

    errorbars = []
    for prob in probs:
        errbar1 = dict(x=epsilons, y=[], yerr=[], label=f"group 1 (p={prob:.4f})", marker='o')
        errbar0 = dict(x=epsilons, y=[], yerr=[], label=f"group 0 (p={1 - prob:.4f})", marker="^")
        errorbars.extend([errbar1, errbar0])
        for epsilon in epsilons:
            errs1, errs0 = [], []
            for seed in tqdm.tqdm(seeds, desc="seeds"):
                x, y = make_labeled_data(n=n_labeled, d=d, mu=mu, prob=prob, sigma=sigma)

                theta_hat = dp_estimator(x=x, y=y, clipping_norm=clipping_norm, epsilon=epsilon)

                x1, y1 = make_labeled_data(n=n_test, d=d, mu=mu, prob=prob, sigma=sigma, group_id=1)
                err1 = compute_accuracy(estimator=theta_hat, x=x1, y=y1)

                # Minority.
                x0, y0 = make_labeled_data(n=n_test, d=d, mu=mu, prob=prob, sigma=sigma, group_id=0)
                err0 = compute_accuracy(estimator=theta_hat, x=x0, y=y0)

                errs1.append(err1)
                errs0.append(err0)

            avg1, std1 = np.mean(errs1), np.std(errs1)
            avg0, std0 = np.mean(errs0), np.std(errs0)
            errbar1['y'].append(avg1)
            errbar1['yerr'].append(std1)
            errbar0['y'].append(avg0)
            errbar0['yerr'].append(std0)

    utils.plot_wrapper(
        errorbars=errorbars,
        options=dict(xlabel="$\epsilon$", xscale="log", yscale='log')
    )


def entropy_sharpening(theta, x, lr=1e-3, num_updates=100):
    theta = theta.clone().requires_grad_(True)
    optimizer = torch.optim.Adam(params=(theta,), lr=lr)
    for _ in range(num_updates):
        optimizer.zero_grad()
        loss = compute_entropy(theta, x)
        loss = loss.mean(dim=0)
        loss.backward()
        assert not torch.isnan(loss).item()
        optimizer.step()
    return theta.detach().clone()


def compute_entropy(theta, x):
    # x: (batch_size, d).
    # theta: (d,).
    margin = (x * theta[None, :]).sum(dim=1)
    # `torch.special.expit` seems amazingly numerically stable!!!
    term1 = - torch.special.expit(margin) * margin
    term2 = F.softplus(margin)
    entropy = term1 + term2
    return entropy


def self_training_vary_n_u(
    alpha=0, beta=1, img_dir=None, epsilon=2., n_us=(1, 5, 10, 30, 100, 300, 1000, 3000, 10000, 30000), **kwargs
):
    probs = (0.5,)
    d = 3
    # Should not use a fixed random \mu; either randomize \mu, or use one with suitable norm.
    # mu = torch.randn((1, d))
    mu = torch.full(size=(1, d), fill_value=1.)
    sigma = 0.8
    n_labeled = 20
    n_test = 10000
    clipping_norm = 5.6  # Let this be the max norm.
    seeds = list(range(1000))

    errorbars = []
    aligns = []
    for prob in probs:
        errbar0 = dict(x=n_us, y=[], yerr=[], label="w/  unlabeled", marker="^", alpha=0.8, capsize=10)
        errbar1 = dict(x=n_us, y=[], yerr=[], label="w/o unlabeled", marker="o", alpha=0.8, capsize=10)
        errbar_opt = dict(x=n_us, y=[], yerr=[], label="optimal", marker='x', alpha=0.8, capsize=10)
        errorbars.extend([errbar0, errbar1, errbar_opt])

        align0 = dict(x=n_us, y=[], yerr=[], label="w/  unlabeled", marker="^", alpha=0.8, capsize=10)
        align1 = dict(x=n_us, y=[], yerr=[], label="w/o unlabeled", marker="o", alpha=0.8, capsize=10)
        align_opt = dict(x=n_us, y=[], yerr=[], label="optimal", marker='x', alpha=0.8, capsize=10)
        aligns.extend([align0, align1, align_opt])

        for n_u in tqdm.tqdm(n_us, desc="n_u"):
            errs1, errs0, errs_opt = [], [], []
            alis1, alis0, alis_opt = [], [], []
            for seed in seeds:
                x, y = make_labeled_data(n=n_labeled, d=d, mu=mu, prob=prob, sigma=sigma)
                x_test, y_test = make_labeled_data(n=n_test, d=d, mu=mu, prob=prob, sigma=sigma)

                max_norm = x.norm(2, dim=1).max().item()
                if max_norm > clipping_norm:
                    print(f'max_norm: {max_norm}, clipping_norm: {clipping_norm}')
                # -- correct

                theta_hat = dp_estimator(x=x, y=y, clipping_norm=clipping_norm, epsilon=epsilon)

                # Without.
                err1 = compute_accuracy(estimator=theta_hat, x=x_test, y=y_test, _print=False)
                errs1.append(err1)

                # With unlabeled.
                x_unlabeled = make_unlabeled_data(n=n_u, d=d, mu=mu, prob=prob, sigma=sigma)
                y_unlabeled = classify(estimator=theta_hat, x=x_unlabeled)  # Pseudo-label.
                theta_tilde = usual_estimator(x=x_unlabeled, y=y_unlabeled)

                theta_bar = alpha * theta_hat + beta * theta_tilde
                err0 = compute_accuracy(estimator=theta_bar, x=x_test, y=y_test, _print=False)
                errs0.append(err0)

                # Optimal classifier.
                err_opt = compute_accuracy(estimator=mu, x=x_test, y=y_test, _print=False)
                errs_opt.append(err_opt)

                def comp_align(est):
                    return (est * mu).sum() / est.norm(2)

                ali1 = comp_align(est=theta_hat)
                ali0 = comp_align(est=theta_tilde)
                ali_opt = comp_align(est=mu)
                alis1.append(ali1)
                alis0.append(ali0)
                alis_opt.append(ali_opt)

            avg0, std0 = np.mean(errs0), np.std(errs0)
            avg1, std1 = np.mean(errs1), np.std(errs1)
            avg_opt, std_opt = np.mean(errs_opt), np.std(errs_opt)

            errbar0['y'].append(avg0)
            errbar0['yerr'].append(std0)
            errbar1['y'].append(avg1)
            errbar1['yerr'].append(std1)
            errbar_opt['y'].append(avg_opt)
            errbar_opt['yerr'].append(std_opt)

            avg0, std0 = np.mean(alis0), np.std(alis0)
            avg1, std1 = np.mean(alis1), np.std(alis1)
            avg_opt, std_opt = np.mean(alis_opt), np.std(alis_opt)

            align0['y'].append(avg0)
            align0['yerr'].append(std0)
            align1['y'].append(avg1)
            align1['yerr'].append(std1)
            align_opt['y'].append(avg_opt)
            align_opt['yerr'].append(std_opt)

    if img_dir is None:
        utils.plot_wrapper(
            errorbars=errorbars,
            options=dict(
                xlabel="$n_u$", xscale="log", yscale='linear',
                ylabel=f"$1 - \mathrm{{err}}$",
                title=f"$\\alpha={alpha}, \\beta={beta}, d={d}, n_l={n_labeled}, \epsilon={epsilon}, \sigma={sigma}, "
                      f"\|\| \mu \|\|_2={mu.norm().item():.4f}$"),
        )
        utils.plot_wrapper(
            errorbars=aligns,
            options=dict(
                xlabel="$n_u$", xscale="log", yscale='linear',
                ylabel=f"$\\frac{{ \mu^\\top \\theta }}{{ \|\| \\theta \|\|_2 }} $",
                title=f"$\\alpha={alpha}, \\beta={beta}, d={d}, n_l={n_labeled}, \epsilon={epsilon}, \sigma={sigma}, "
                      f"\|\| \mu \|\|_2={mu.norm().item():.4f}$"),
        )
    else:
        alpha_str = utils.float2str(alpha)
        beta_str = utils.float2str(beta)

        img_path = utils.join(img_dir, f'vary_n_u_alpha_{alpha_str}_beta_{beta_str}_d_{d}_err')
        utils.plot_wrapper(
            errorbars=errorbars,
            suffixes=('.png', '.pdf'),
            options=dict(
                xlabel="$n_u$", xscale="log", yscale='linear',
                ylabel=f"$1 - \mathrm{{err}}$",
                title=f"$\\alpha={alpha}, \\beta={beta}, d={d}, n_l={n_labeled}, \epsilon={epsilon}, \sigma={sigma}, "
                      f"\|\| \mu \|\|_2={mu.norm().item():.4f}$"),
            img_path=img_path,
        )

        img_path = utils.join(img_dir, f'vary_n_u_alpha_{alpha_str}_beta_{beta_str}_d_{d}_alignment')
        utils.plot_wrapper(
            img_path=img_path,
            suffixes=('.png', '.pdf'),
            errorbars=aligns,
            options=dict(
                xlabel="$n_u$", xscale="log", yscale='linear',
                ylabel=f"$\\frac{{ \mu^\\top \\theta }}{{ \|\| \\theta \|\|_2 }} $",
                title=f"$\\alpha={alpha}, \\beta={beta}, d={d}, n_l={n_labeled}, \epsilon={epsilon}, \sigma={sigma}, "
                      f"\|\| \mu \|\|_2={mu.norm().item():.4f}$"),
        )


def self_training(alpha=0, beta=1, img_dir=None, **kwargs):
    """Compare error with and without unlabeled data.

    Semi-sup estimator = alpha * private estimator + beta * PL estimator.

    Questions:
        - separation between overparam vs underparam?
        - phase transition between low to high privacy?
        - how does noise variance in the data model affect things?
    """

    epsilons = (0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2.0)
    probs = (0.5,)
    d = 3
    # Should not use a fixed random \mu; either randomize \mu, or use one with suitable norm.
    # mu = torch.randn((1, d))
    mu = torch.full(size=(1, d), fill_value=1.)
    sigma = 1
    n_labeled = 30
    n_unlabeled = 30000  # x100 factor.
    n_test = 10000
    clipping_norm = 5.5  # Let this be the max norm.
    seeds = list(range(500))

    errorbars = []
    aligns = []
    for prob in probs:
        errbar0 = dict(x=epsilons, y=[], yerr=[], label="w/  unlabeled", marker="^", alpha=0.8, capsize=10)
        errbar1 = dict(x=epsilons, y=[], yerr=[], label="w/o unlabeled", marker="o", alpha=0.8, capsize=10)
        errbar_opt = dict(x=epsilons, y=[], yerr=[], label="optimal", marker='x', alpha=0.8, capsize=10)
        errorbars.extend([errbar0, errbar1, errbar_opt])

        align0 = dict(x=epsilons, y=[], yerr=[], label="w/  unlabeled", marker="^", alpha=0.8, capsize=10)
        align1 = dict(x=epsilons, y=[], yerr=[], label="w/o unlabeled", marker="o", alpha=0.8, capsize=10)
        align_opt = dict(x=epsilons, y=[], yerr=[], label="optimal", marker='x', alpha=0.8, capsize=10)
        aligns.extend([align0, align1, align_opt])
        # -- correct

        for epsilon in tqdm.tqdm(epsilons, desc="epsilon"):
            errs1, errs0, errs_opt = [], [], []
            alis1, alis0, alis_opt = [], [], []
            for seed in seeds:
                x, y = make_labeled_data(n=n_labeled, d=d, mu=mu, prob=prob, sigma=sigma)
                x_test, y_test = make_labeled_data(n=n_test, d=d, mu=mu, prob=prob, sigma=sigma)

                max_norm = x.norm(2, dim=1).max().item()
                if max_norm > clipping_norm:
                    print(f'max_norm: {max_norm}, clipping_norm: {clipping_norm}')
                # -- correct

                theta_hat = dp_estimator(x=x, y=y, clipping_norm=clipping_norm, epsilon=epsilon)

                # Without.
                err1 = compute_accuracy(estimator=theta_hat, x=x_test, y=y_test, _print=False)
                errs1.append(err1)

                # With unlabeled.
                x_unlabeled = make_unlabeled_data(n=n_unlabeled, d=d, mu=mu, prob=prob, sigma=sigma)
                y_unlabeled = classify(estimator=theta_hat, x=x_unlabeled)  # Pseudo-label.
                theta_tilde = usual_estimator(x=x_unlabeled, y=y_unlabeled)

                theta_bar = alpha * theta_hat + beta * theta_tilde
                err0 = compute_accuracy(estimator=theta_bar, x=x_test, y=y_test, _print=False)
                errs0.append(err0)

                # Optimal classifier.
                err_opt = compute_accuracy(estimator=mu, x=x_test, y=y_test, _print=False)
                errs_opt.append(err_opt)

                def comp_align(est):
                    return (est * mu).sum() / est.norm(2)

                ali1 = comp_align(est=theta_hat)
                ali0 = comp_align(est=theta_tilde)
                ali_opt = comp_align(est=mu)
                alis1.append(ali1)
                alis0.append(ali0)
                alis_opt.append(ali_opt)

            avg0, std0 = np.mean(errs0), np.std(errs0)
            avg1, std1 = np.mean(errs1), np.std(errs1)
            avg_opt, std_opt = np.mean(errs_opt), np.std(errs_opt)

            errbar0['y'].append(avg0)
            errbar0['yerr'].append(std0)
            errbar1['y'].append(avg1)
            errbar1['yerr'].append(std1)
            errbar_opt['y'].append(avg_opt)
            errbar_opt['yerr'].append(std_opt)

            avg0, std0 = np.mean(alis0), np.std(alis0)
            avg1, std1 = np.mean(alis1), np.std(alis1)
            avg_opt, std_opt = np.mean(alis_opt), np.std(alis_opt)

            align0['y'].append(avg0)
            align0['yerr'].append(std0)
            align1['y'].append(avg1)
            align1['yerr'].append(std1)
            align_opt['y'].append(avg_opt)
            align_opt['yerr'].append(std_opt)

    if img_dir is None:
        utils.plot_wrapper(
            errorbars=errorbars,
            options=dict(
                xlabel="$\epsilon$", xscale="linear", yscale='linear',
                ylabel=f"$1 - \mathrm{{err}}$",
                title=f"$\\alpha={alpha}, \\beta={beta}, d={d}, n_l={n_labeled}, n_u={n_unlabeled}, \sigma={sigma}, "
                      f"\|\| \mu \|\|_2={mu.norm().item():.4f}$"),
        )
        utils.plot_wrapper(
            errorbars=aligns,
            options=dict(
                xlabel="$\epsilon$", xscale="linear", yscale='linear',
                ylabel=f"$\\frac{{ \mu^\\top \\theta }}{{ \|\| \\theta \|\|_2 }} $",
                title=f"$\\alpha={alpha}, \\beta={beta}, d={d}, n_l={n_labeled}, n_u={n_unlabeled}, \sigma={sigma}, "
                      f"\|\| \mu \|\|_2={mu.norm().item():.4f}$"),
        )
    else:
        alpha_str = utils.float2str(alpha)
        beta_str = utils.float2str(beta)

        img_path = utils.join(img_dir, f'alpha_{alpha_str}_beta_{beta_str}_d_{d}_err')
        utils.plot_wrapper(
            errorbars=errorbars,
            suffixes=('.png', '.pdf'),
            options=dict(
                xlabel="$\epsilon$", xscale="linear", yscale='linear',
                ylabel=f"$1 - \mathrm{{err}}$",
                title=f"$\\alpha={alpha}, \\beta={beta}, d={d}, n_l={n_labeled}, n_u={n_unlabeled}, \sigma={sigma}, "
                      f"\|\| \mu \|\|_2={mu.norm().item():.4f}$"),
            img_path=img_path,
        )

        img_path = utils.join(img_dir, f'alpha_{alpha_str}_beta_{beta_str}_d_{d}_alignment')
        utils.plot_wrapper(
            img_path=img_path,
            suffixes=('.png', '.pdf'),
            errorbars=aligns,
            options=dict(
                xlabel="$\epsilon$", xscale="linear", yscale='linear',
                ylabel=f"$\\frac{{ \mu^\\top \\theta }}{{ \|\| \\theta \|\|_2 }} $",
                title=f"$\\alpha={alpha}, \\beta={beta}, d={d}, n_l={n_labeled}, n_u={n_unlabeled}, \sigma={sigma}, "
                      f"\|\| \mu \|\|_2={mu.norm().item():.4f}$"),
        )


def sweep_self_training(pairs=None, **kwargs):
    if pairs is None:
        N = 5
        pairs = ((i / N, 1. - i / N) for i in range(N + 1))  # (alpha, beta).
    img_dir = "./self_training"
    for alpha, beta in pairs:
        self_training(alpha=alpha, beta=beta, img_dir=img_dir)


def main(task="self_training", seed=0, **kwargs):
    torch.set_default_dtype(torch.float64)

    if task == "fairness":
        fairness(**kwargs)
    elif task == "self_training":
        # python gaussian_model.py --task "self_training"
        self_training(**kwargs)
    elif task == "self_training_vary_n_u":
        # python gaussian_model.py --task "self_training_vary_n_u" --img_dir "./self_training/"
        self_training_vary_n_u(**kwargs)
    elif task == "sweep_self_training":
        # python gaussian_model.py --task "sweep_self_training"
        # Make plots, each with a fixed interpolation coefficient between original private and self-supervised.
        sweep_self_training(**kwargs)
    else:
        raise ValueError


if __name__ == "__main__":
    fire.Fire(main)
