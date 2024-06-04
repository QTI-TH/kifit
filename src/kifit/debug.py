import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from kifit.loadelems import Elem
from kifit.performfit import compute_ll, generate_element_sample

elem = Elem.get('Ybmin')

n = 3
nsamples = 1000
alphas, lls = [], []
colors = sns.color_palette("magma", n_colors=n).as_hex()

for i in range(n):
    np.random.seed(42*(i+1))
    nsamples = 10*(i+1)
    print(f"nsamples: {nsamples}")
    elemsamples = generate_element_sample(elem, nsamples)
    alphasample = np.linspace(-5e-8, 1e-8, nsamples)
    one_alphas, one_lls = compute_ll(elem, alphasample, nelemsamples=nsamples, elementsamples=elemsamples)
    alphas.append(one_alphas)
    lls.append(one_lls)

plt.figure(figsize=(6, 6*6/8))
for i in range(n):
    plt.plot(alphas[i], lls[i], label=f"Sample {i}", color=colors[i], alpha=0.7, lw=1)
plt.legend()
plt.savefig("debug.png", dpi=600)
plt.show()