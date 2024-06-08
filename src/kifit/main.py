from kifit.loadelems import Elem
from kifit.performfit import sample_alphaNP_fit
from kifit.plotfit import plot_linfit, plot_alphaNP_ll, plot_mphi_alphaNP

ca = Elem.get('Ybmin')
print(ca.get_dimensions)
print()
print("relative uncertainties")
print(ca.sig_nu / ca.nu)
print(ca.sig_m_a_in / ca.m_a_in)
print(ca.sig_m_ap_in / ca.m_ap_in)

plot_linfit(ca, resmagnifac=1)

num_samples_det = 100
num_elemsamples_search = 100  # 200
num_alphasamples_search = 100
num_elemsamples_experiment = 100
num_alphasamples_experiment = 100
num_experiments = 5
num_blocks = 1

# search hyper-parameters
max_iter = 20
scalefactor = 0.3
# sig_new_alpha_fraction = 0.12


gkp_dims = []
nmgkp_dims = []

mc_output = sample_alphaNP_fit(
    ca,
    nelemsamples_search=num_elemsamples_search,
    nalphasamples_search=num_alphasamples_search,
    nexps=num_experiments,
    nelemsamples_exp=num_elemsamples_experiment,
    nalphasamples_exp=num_alphasamples_experiment,
    nblocks=num_blocks,
    maxiter=max_iter,
    mphivar=False,
    plot_output=True,
    scalefactor=scalefactor,
    sigalphainit=1,
    # sig_new_alpha_fraction=sig_new_alpha_fraction,
    x0=0,
)

# plot_alphaNP_ll(ca, mc_output, xind=0, ylims=[-1e7, 2e8])

# plot_mphi_alphaNP(ca, mc_output, gkp_dims, nmgkp_dims, num_samples_det,
#     showalldetbounds=True, showallowedfitpts=True)
