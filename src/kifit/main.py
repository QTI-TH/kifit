import matplotlib.pyplot as plt
import numpy as np

from kifit import performfit
from kifit.loadelems import Elem
from kifit.plotfit import plot_loss_varying_alphaNP

nsamples = 20000
elem = Elem.get("Ca_WT_Aarhus_PTB_2021")
elem.sig_alphaNP_init = 1e-9

# elem._update_Xcoeffs(400)

element_samples = performfit.generate_element_sample(elem, nsamples)
alphaNP_list, ll_list, ba, _, best_alpha = performfit.iterative_mc_search(
    elem=elem, element_samples=element_samples, niter=5, delta_alpha_ratio=0.4
)
plot_loss_varying_alphaNP(alphaNP_list=alphaNP_list, ll_list=ll_list)

print(f"\n\nBest alpha value: {ba}")
