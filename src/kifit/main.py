import matplotlib.pyplot as plt
import numpy as np

from kifit import performfit
from kifit.loadelems import Elem

nsamples = 1000
elem = Elem.get("Yb")

alphaNP_list, ll_list = performfit.compute_sample_ll(
    elem=elem, nsamples=nsamples, save_sample=False
)

plt.scatter(alphaNP_list, ll_list)
plt.show()
