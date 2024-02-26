import matplotlib.pyplot as plt
import numpy as np

from kifit import performfit
from kifit.loadelems import Elem
from kifit.plotfit import plot_loss_varying_alphaNP

nsamples = 2000
elem = Elem.get("Yb")

element_samples = performfit.generate_element_sample(elem, nsamples)
alphaNP_list, ll_list, alpha_best, ll_best = performfit.iterative_mc_search(
    elem=elem, element_samples=element_samples, niter=5, decay_rate=0.3
)
plot_loss_varying_alphaNP(alphaNP_list=alphaNP_list, ll_list=ll_list)

print(alpha_best, ll_best)
