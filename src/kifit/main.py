from kifit.loadelems import Elem
from kifit.performfit import sample_alphaNP_fit
from kifit.plotfit import plot_linfit, plot_alphaNP_ll, plot_mphi_alphaNP

ca = Elem.get('Ca')
print(ca.get_dimensions)

plot_linfit(ca, resmagnifac=1)

num_samples = 500
max_iter = 10
n_blocks = 10

gkp_dims = [3]
nmgkp_dims = [3]

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

(best_alpha_parabola_list, sig_best_alpha_parabola_list,
    best_alpha_pt_list, sig_best_alpha_pt_list,
    lb_list, sig_lb_list, ub_list, sig_ub_list) = sample_alphaNP_fit(
        ca, 
        nsamples_search=num_samples, 
        nexps=100,
        nsamples_exp=100,
        nblocks=n_blocks, 
        maxiter=max_iter,
        mphivar=False, 
        draw_output=False)

# print("best_alpha_parabola_list", best_alpha_parabola_list)
# print("sig_best_alpha_parabola_list", sig_best_alpha_parabola_list)

# bring back
# plot_alphaNP_ll(ca, alphalist, llist)
#
# plot_mphi_alphaNP(ca, alphalist, llist, gkp_dims, nmgkp_dims, num_samples,
#     showalldetbounds=True, showallowedfitpts=True)
