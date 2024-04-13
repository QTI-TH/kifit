from kifit.loadelems import Elem
from kifit.performfit import (
    iterative_mc_search
    )
from kifit.plotfit import plot_linfit, plot_alphaNP_ll, plot_mphi_alphaNP

ca = Elem.get('Ca')
ca.sig_alphaNP_init = 1e-8

ca._update_Xcoeffs(10)

result = iterative_mc_search(
    elem=ca,
    n_sampled_elems=5000,
    mphivar=False,
    niter=10,
    delta_alpha_ratio=0.99
)