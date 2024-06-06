import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from kifit.loadelems import Elem
from kifit.performfit import compute_ll, generate_element_sample

elem = Elem.get('Ybmin')

n = 4
nsamples_list = [103, 503, 1003, 1503]
alphas, lls = [], []
colors = ["orange", "red", "royalblue", "green"]

for i in range(n):
    np.random.seed(42*(i+1))
    nsamples = nsamples_list[i]
    print(f"nsamples: {nsamples}")
    elemsamples = generate_element_sample(elem, nsamples)
    alphasample = np.linspace(-3e-8, 0, nsamples)
    one_alphas, one_lls = compute_ll(
        elem, 
        alphasample, 
        nelemsamples=nsamples, 
        elementsamples=elemsamples,
        decomposition_method="spectral"
    )
    alphas.append(one_alphas)
    lls.append(one_lls)
    np.save(arr=np.array(one_alphas), file=f"debug_data/alphas_{nsamples}n.npy")
    np.save(arr=np.array(one_lls), file=f"debug_data/lls_{nsamples}n.npy")

plt.figure(figsize=(6, 6*6/8))
for i in range(n):
    plt.plot(alphas[i], lls[i], label=f"n = {nsamples_list[i]}", color=colors[i], alpha=0.7, lw=1)
    min_index = np.argmin(lls[i])
    print(alphas[i][min_index])
    plt.vlines(alphas[i][min_index], min([min(l) for l in lls]), max([max(l) for l in lls]), ls="--", lw=1, color=colors[i])
plt.legend()
plt.savefig(f"debug_{nsamples}_spec.png", dpi=600)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"LL")

