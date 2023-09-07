import numpy as np
import matplotlib.pyplot as plt


def plot_linear_fits(slopes, intercepts, data, target_index=0):
    """Plot linear fit after simple linear regression optimization."""
    nplots = len(slopes)
    xaxis = data.T[0]
    xaxis = (xaxis - np.min(xaxis)) / (np.max(xaxis) - np.min(xaxis))
    # list of indices without the target one
    idx = np.delete(np.arange(len(slopes) + 1), target_index)

    x = np.linspace(0, 1, 100)

    plt.figure(figsize=(7 * nplots, 5))
    for i in range(nplots):
        plt.subplot(1, nplots, i + 1)
        plt.title(f"Fit for columns {target_index}{idx[i]}")
        plt.plot(
            x, intercepts[i] + slopes[i] * x, lw=1.5, alpha=0.7, color="black", ls="-"
        )
        plt.scatter(
            xaxis, intercepts[i] + slopes[i] * xaxis, s=80, alpha=0.5, color="purple"
        )
        plt.grid(True)
        plt.xlabel("Normalised x axis")
        plt.ylabel(f"{target_index}{idx[i]} transition")
    plt.tight_layout()
    plt.savefig("linear_fits.pdf")


def histogram_parameter(parameter_stat, parameter_index):
    """Plot histogram of optimized parameter."""

    estimates = np.asarray(parameter_stat).T[parameter_index]
    mean = np.mean(estimates)

    plt.figure(figsize=(4.5, 4.5 * 6 / 8))
    plt.hist(estimates, bins=30, histtype="step", color="royalblue")
    plt.hist(
        estimates,
        bins=30,
        histtype="stepfilled",
        edgecolor="royalblue",
        color="royalblue",
        alpha=0.3,
        hatch="//",
    )
    plt.vlines(
        mean,
        0,
        len(estimates) / 4,
        color="black",
        label=rf"$\langle p[{parameter_index}] \rangle$: {mean:.4}",
    )
    plt.legend(framealpha=1)
    plt.xlabel(f"p[{parameter_index}]")
    plt.ylabel("#")
    plt.savefig("cma_hist.png", bbox_inches="tight")
