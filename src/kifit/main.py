from kifit.loadelems import Elem
from kifit.performfit import sample_alphaNP_fit
from kifit.plotfit import plot_linfit, plot_alphaNP_ll, plot_mphi_alphaNP

ca = Elem.get('Ybmin')
print(ca.get_dimensions)

plot_linfit(ca, resmagnifac=1)

num_samples_det = 100
num_samples_search = 200
num_samples_experiment = 400
num_experiments = 5
num_blocks = 1

# search hyper-parameters
max_iter = 100
scalefactor = 0.3
# sig_new_alpha_fraction = 0.12


gkp_dims = []
nmgkp_dims = []

mc_output = sample_alphaNP_fit(
    ca,
    nsamples_search=num_samples_search,
    nexps=num_experiments,
    nsamples_exp=num_samples_experiment,
    nblocks=num_blocks,
    maxiter=max_iter,
    mphivar=False,
    plot_output=True,
    scalefactor=scalefactor,
    sigalphainit=1.,
    # sig_new_alpha_fraction=sig_new_alpha_fraction,
    x0=0,
)

plot_alphaNP_ll(ca, mc_output, xind=0)

# plot_mphi_alphaNP(ca, mc_output, gkp_dims, nmgkp_dims, num_samples_det,
#     showalldetbounds=True, showallowedfitpts=True)
