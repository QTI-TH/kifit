from kifit.loadelems import Elem
from kifit.performfit_new import sample_alphaNP_fit, generate_path
from kifit.plotfit import plot_linfit, plot_alphaNP_ll, plot_mphi_alphaNP

# load element data
datafile = "Yb_PTB_2024"

elem = Elem.get(datafile)

gkp_dims = [3]
nmgkp_dims = []
elem.check_det_dims(gkp_dims, nmgkp_dims)


# elem.alphaNP_GKP(ainds=[0, 1, 2], iinds=[0, 1])
num_samples_det = 100
num_searches = 10
num_elemsamples_search = 100  # 200

num_experiments = 12
num_elemsamples_experiment = 100
num_alphasamples_experiment = 100
block_size = 10

# search hyper-parameters
max_iter = 1000
opt_method = "differential_evolution"

# define output folder's name

# some initial prints
elem.print_dimensions
elem.print_relative_uncertainties

plot_linfit(elem, resmagnifac=1)

mc_output = sample_alphaNP_fit(
    elem,
    nsearches=num_searches,
    nelemsamples_search=num_elemsamples_search,
    nexps=num_experiments,
    nelemsamples_exp=num_elemsamples_experiment,
    nalphasamples_exp=num_alphasamples_experiment,
    block_size=block_size,
    maxiter=max_iter,
    mphivar=False,
    plot_output=True,
    opt_method=opt_method,
    x0=0,
    min_percentile=1,
)

plot_alphaNP_ll(elem, mc_output,
    xind=0,
    gkpdims=gkp_dims, nmgkpdims=nmgkp_dims,
    ndetsamples=num_samples_det,
    showalldetbounds=True, showbestdetbounds=True)

# plot_mphi_alphaNP(elem, mc_output, gkp_dims, nmgkp_dims, num_samples_det,
#     showalldetbounds=True, showallowedfitpts=True)
