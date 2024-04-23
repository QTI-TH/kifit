from kifit.loadelems import Elem
from kifit.performfit import sample_alphaNP_fit
from kifit.plotfit import plot_linfit, plot_alphaNP_ll, plot_mphi_alphaNP

ca = Elem.get('Camin')
print(ca.get_dimensions)

plot_linfit(ca, resmagnifac=1)

# num_samples_det = 100
num_samples_det = 100
num_samples_search = 100
num_samples_experiment = 200
num_experiments = 10
num_blocks = 5
sig_new_alpha_fraction = 0.3

max_iter = 100

gkp_dims = []
nmgkp_dims = []

mc_output = sample_alphaNP_fit(
    ca,
    nsamples_search=num_samples_search,
    nexps=num_experiments, 
    nsamples_exp=num_samples_experiment,
    nblocks=num_blocks, 
    maxiter=max_iter,
    mphivar=True, 
    plot_output=False,
    sig_new_alpha_fraction=sig_new_alpha_fraction,
)

plot_alphaNP_ll(ca, mc_output, xind=0)

# plot_mphi_alphaNP(ca, mc_output, gkp_dims, nmgkp_dims, num_samples_det,
#     showalldetbounds=True, showallowedfitpts=True)
