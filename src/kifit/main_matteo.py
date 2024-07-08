import datetime
from kifit.loadelems import Elem
from kifit.performfit_new import sample_alphaNP_fit, generate_path
from kifit.plotfit import plot_linfit, plot_alphaNP_ll

# load element data
datafile = "Pippo"

element_collection = [Elem.get("Camin"), Elem.get("Ybmin")]

print(element_collection[0].alphaNP_init)
print(element_collection[0].sig_alphaNP_init)

num_samples_det = 100
num_searches = 5
num_elemsamples_search = 100 

num_experiments = 2
num_elemsamples_experiment = 100  
num_alphasamples_experiment = 100  
num_blocks = 1

# search hyper-parameters
max_iter = 500
opt_method = "Powell"

# define output folder's name
output_filename = f"{datafile}_{opt_method}_{num_searches}searches_{num_experiments}nexps_{num_elemsamples_search}es_{num_elemsamples_experiment}ee_{num_alphasamples_experiment}ae"
_, plot_path = generate_path(output_filename)



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

