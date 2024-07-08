import datetime
from kifit.loadelems import Elem
from kifit.performfit_new import sample_alphaNP_fit, generate_path
from kifit.plotfit import plot_linfit, plot_alphaNP_ll

# load element data
datafile = "Yb_Ca"

element_collection = [Elem.get("Camin")]

# TODO: this is broken for list of elements
# gkp_dims = [3, 4]
# nmgkp_dims = [3, 4]

# for elem in element_collection:
#     elem.check_det_dims(gkp_dims, nmgkp_dims)

# elem.alphaNP_GKP(ainds=[0, 1, 2], iinds=[0, 1])
num_samples_det = 200
num_searches = 10
num_elemsamples_search = 200 

num_experiments = 5  
num_elemsamples_experiment = 100  
num_alphasamples_experiment = 100  
num_blocks = 1

# search hyper-parameters
max_iter = 1000
opt_method = "Powell"

# define output folder's name
output_filename = f"{datafile}_{opt_method}_{num_searches}searches_{num_experiments}nexps_{num_elemsamples_search}es_{num_elemsamples_experiment}ee_{num_alphasamples_experiment}ae"
_, plot_path = generate_path(output_filename)


# plot_linfit(elem, resmagnifac=1, plot_path=plot_path)

mc_output = sample_alphaNP_fit(
    element_collection,
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

# plot_alphaNP_ll(
#     elem, 
#     output_filename,
#     mc_output,
#     xind=0,
#     gkpdims=gkp_dims, 
#     nmgkpdims=nmgkp_dims,
#     ndetsamples=num_samples_det,
#     showalldetbounds=True, 
#     showbestdetbounds=True
# )

