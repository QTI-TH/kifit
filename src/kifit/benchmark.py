import numpy as np
import matplotlib.pyplot as plt
from plots import histogram_parameter


data = []
means = []
colors = ["orange", "red", "blue"]
for i in range(3):
    data.append(np.load(f"hist_{i+1}.npy"))
    means.append(np.mean(data[-1]))

plt.figure(figsize=(7.5, 7.5 * 5 / 8))
for i in range(3):
    estimates = data[i]
    mean = means[i]
    plt.hist(estimates, bins=30, histtype="step", color=colors[i])
    plt.hist(
        estimates,
        bins=30,
        histtype="stepfilled",
        edgecolor=colors[i],
        color=colors[i],
        alpha=0.3,
        hatch="//",
    )
    plt.vlines(
        mean,
        0,
        len(estimates) / 4,
        color=colors[i],
        label=rf"$\langle p[{-1}] \rangle$: {mean:.4}",
    )
    plt.legend(framealpha=1)
    plt.xlabel(f"p[{-1}]")
    plt.ylabel("#")
plt.savefig("cma_hist.png", bbox_inches="tight")
