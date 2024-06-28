from kifit.loadelems import Elem
from kifit.performfit_new import sample_alphaNP_fit
from kifit.plotfit import plot_linfit, plot_alphaNP_ll, plot_mphi_alphaNP

elem = Elem.get('Ybmin')
print(elem.get_dimensions)
print()
print("relative uncertainties")
print(elem.sig_nu / elem.nu)
print(elem.sig_m_a_in / elem.m_a_in)
print(elem.sig_m_ap_in / elem.m_ap_in)

plot_linfit(elem, resmagnifac=1)

gkp_dims = [3, 4]
nmgkp_dims = [3, 4]
elem.check_det_dims(gkp_dims, nmgkp_dims)


# elem.alphaNP_GKP(ainds=[0, 1, 2], iinds=[0, 1])
num_samples_det = 100

num_searches = 5
num_elemsamples_search = 100  # 200

num_experiments = 1  #5
num_elemsamples_experiment = 10  #100
num_alphasamples_experiment = 10  #100
num_blocks = 1

# search hyper-parameters
max_iter = 100

mc_output = sample_alphaNP_fit(
    elem,
    nsearches=num_searches,
    nelemsamples_search=num_elemsamples_search,
    nexps=num_experiments,
    nelemsamples_exp=num_elemsamples_experiment,
    nalphasamples_exp=num_alphasamples_experiment,
    nblocks=num_blocks,
    maxiter=max_iter,
    mphivar=False,
    plot_output=True,
    x0=0,
)

plot_alphaNP_ll(elem, mc_output,
    xind=0,
    gkpdims=gkp_dims, nmgkpdims=nmgkp_dims,
    ndetsamples=num_samples_det,
    showalldetbounds=True, showbestdetbounds=True)
