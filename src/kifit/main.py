from kifit.loadelems import Elem
from kifit.performfit import sample_alphaNP_fit
from kifit.plotfit import plot_linfit, plot_alphaNP_ll, plot_mphi_alphaNP

ca = Elem.get('Camin')
print(ca.get_dimensions)

plot_linfit(ca, resmagnifac=1)

num_samples = 50
max_iter = 10
n_blocks = 10

gkp_dims = [3]
nmgkp_dims = []

#
# # FIT
# # Varying alphaNP only
# # ca.set_alphaNP_init(0, sigalpha=1e-8)
# alphalist, llist = sample_alphaNP_fit_fixed_elemparams(ca, num_samples, mphivar=True)
#
# plot_alphaNP_ll(ca, alphalist, llist,
#     plotname=("alphaNPvar_" + str(num_samples)),
#     xlims=[-1e-7, 1e-7])
#
# plot_mphi_alphaNP(ca, alphalist, llist, gkp_dims, nmgkp_dims, num_samples)
#
# # Varying elem data and alphaNP
# # ca.set_alphaNP_init(min(ca.alphaNP_GKP(3), key=abs), 1e-5)
# alphalist, llist = sample_alphaNP_fit_fixed_elemparams(
#     ca, num_samples, mphivar=True)
#
# plot_alphaNP_ll(ca, alphalist, llist, plotname="fixed_elemparams")
#
# plot_mphi_alphaNP(ca, alphalist, llist, gkp_dims, nmgkp_dims, num_samples,
#     plotname="fixed_elemparams",
#     showalldetbounds=True, showallowedfitpts=True)

mc_output = sample_alphaNP_fit(
    ca, nsamples_search=num_samples, nexps=n_blocks, nsamples_exp=num_samples,
    nblocks=n_blocks, maxiter=max_iter,
    mphivar=True, plot_output=False)

plot_alphaNP_ll(ca, mc_output, xind=0)

plot_mphi_alphaNP(ca, mc_output, gkp_dims, nmgkp_dims, num_samples,
    showalldetbounds=True, showallowedfitpts=True)
