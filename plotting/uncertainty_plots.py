from config.config import config
from common.common import uncertainity_estimate
import matplotlib.pyplot as plt


def plot_model(X_obs, y_obs, X_true, y_mean, model, l2, iters=200, n_std=2, ax=None):
    if ax is None:
        plt.close("all")
        plt.clf()
        fig, ax = plt.subplots(1 ,1)
    y_mean, y_std = uncertainity_estimate(X_true, model, iters, l2)

    ax.plot(X_obs, y_obs, ls="none", marker="o", color="0.1", alpha=0.8, label="observed")
    ax.plot(X_true, y_mean, ls="-", color="b", label="mean")
    for i in range(n_std):
        ax.fill_between(
            X_true,
            y_mean - y_std * ((i + 1.)/2.),
            y_mean + y_std * ((i + 1.)/2.),
            color="b",
            alpha=0.1
        )
    ax.legend()
    return ax
