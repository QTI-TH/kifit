import datetime
from kifit.loadelems import Elem
from kifit.performfit_new import sample_alphaNP_fit, generate_path
from kifit.plotfit import plot_linfit, plot_alphaNP_ll

# load element data
datafile = "Ybmin"
elem = Elem.get(datafile)

num_samples_det = 100
num_searches = 10
num_elemsamples_search = 100  # 200

num_experiments = 100
num_elemsamples_experiment = 100
num_alphasamples_experiment = 100
num_blocks = 1

# search hyper-parameters
max_iter = 1000
opt_method = "differential_evolution"

# define output folder's name
output_filename = f"{datafile}_{opt_method}_{num_searches}searches_{num_experiments}nexps_{num_elemsamples_search}es_{num_elemsamples_experiment}ee_{num_alphasamples_experiment}ae"
_, plot_path = generate_path(output_filename)

# some initial prints
print(elem.get_dimensions)
print()
print("relative uncertainties")
print(elem.sig_nu / elem.nu)
print(elem.sig_m_a_in / elem.m_a_in)
print(elem.sig_m_ap_in / elem.m_ap_in)

plot_linfit(elem, resmagnifac=1, plot_path=plot_path)


gkp_dims = []
nmgkp_dims = []

mc_output = sample_alphaNP_fit(
    elem,
    output_filename=output_filename,
    nsearches=num_searches,
    nelemsamples_search=num_elemsamples_search,
    nexps=num_experiments,
    nelemsamples_exp=num_elemsamples_experiment,
    nalphasamples_exp=num_alphasamples_experiment,
    nblocks=num_blocks,
    maxiter=max_iter,
    mphivar=False,
    plot_output=True,
    opt_method=opt_method,
    x0=0,
)

plot_alphaNP_ll(elem, mc_output, xind=0, xlims=[-5e-8, 0], plot_path=plot_path)
