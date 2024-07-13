from kifit.loadelems import Elem
from kifit.performfit import sample_alphaNP_det
from kifit.plotfit import plot_linfit, plot_alphaNP_ll, plot_mphi_alphaNP

elem = Elem.get('Ybmin')

gkp_dim = 3
# nmgkp_dim = 3

# elem.alphaNP_GKP(ainds=[0, 1, 2], iinds=[0, 1])
num_samples_det = 100

gkp_output = sample_alphaNP_det(
    elem=elem,
    dim=gkp_dim,
    nsamples=num_samples_det,
    mphivar=False,
    gkp=True)

# nmgkp_output = sample_alphaNP_det(
#     elem=elem,
#     dim=nmgkp_dim,
#     nsamples=num_samples_det,
#     mphivar=False,
#     gkp=False)
#
