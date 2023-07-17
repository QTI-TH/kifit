import numpy as np
import matplotlib.pyplot as plt


def plot_linear_fits(slopes, intercepts, data, target_index=0):
    """Plot linear fit after simple linear regression optimization."""
    nplots = len(slopes)
    xaxis = data.T[0]
    xaxis = (xaxis - np.min(xaxis)) / (np.max(xaxis) - np.min(xaxis))
    print(xaxis)
    # list of indices without the target one
    idx = np.delete(np.arange(len(slopes) + 1), target_index)
    print(idx)

    x = np.linspace(0, 1, 100)

    plt.figure(figsize=(8 * nplots, 5))
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
    plt.savefig("linear_fits.png")
