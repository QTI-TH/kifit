from kifit.loadelems import Elem
from kifit.performfit import (sample_ll_fixed_elemparams, samplelem,
    get_delchisq_crit)
from kifit.plotfit import draw_mc_output

ca = Elem.get('Yb')

num_samples = 50000

# FIT
# Varying alphaNP only
# ll_alphaNPvar, alphaNPvar_simp = sample_ll_fixed_elemparams(ca, num_samples,
#     mphivar=True)
#
# draw_mc_output(alphaNPvar_simp, ll_alphaNPvar,
#     confints=True, nsigmas=[1, 2], dof=1, xlabel=r"$\alpha_{\mathrm{NP}}$",
#     plotname=("alphaNPvar_" + str(num_samples)))
#

# Varying elem data and alphaNP
ca.set_alphaNP_init(min(ca.alphaNP_GKP(3), key=abs), 1e-5)
alphaNPllist, alphaNP_GKP, alphaNP_NMGKP, elemvar = samplelem(ca, num_samples,
    orthkifit=True, genkifit=[], nmgenkifit=[], mphivar=False)

draw_mc_output(alphaNPllist[0][0], alphaNPllist[0][1],
    confints=True, nsigmas=[1, 2], dof=1, xlabel=r"$\alpha_{\mathrm{NP}}$",
    plotname=("elemalphaNPvar_" + str(num_samples) + "_" + ca.id))
llim = 10 * get_delchisq_crit(2, 1)

draw_mc_output(alphaNPllist[0][0], alphaNPllist[0][1],
    confints=True, nsigmas=[1, 2], dof=1, xlabel=r"$\alpha_{\mathrm{NP}}$",
    xlims=[-1e-6, 1e-6], ylims=[0, llim],
    plotname=("zoom_elemalphaNPvar_" + str(num_samples) + "_" + ca.id))


# llim = 10 * get_delchisq_crit(2, 1)
# draw_mc_output(alphaNPvar_full, ll_elemalphaNPvar,
#     confints=True, nsigmas=[1, 2], dof=1, xlabel=r"$\alpha_{\mathrm{NP}}$",
#     xlims=[-1e-12, 1e-12], ylims=[0, llim],
#     plotname=("zoom_elemalphaNPvar_" + str(num_samples) + "_" + ca.id))
#

###############################################################################

# ca_elem_params = multivariate_normal.rvs(
#     ca.means_input_params,
#     np.diag(ca.stdevs_input_params**2),
#     size=num_samples
# )
# ca_fit_params = multivariate_normal.rvs(
#     ca.means_fit_params,
#     np.diag(ca.stdevs_fit_params**2),
#     size=num_samples
# )
#
# absd_samples = []
#
# for i in range(num_samples):
#     if i % (num_samples // 100) == 0:
#         prog = np.round(i / num_samples * 100, 1)
#         print("Progress", prog, "%")
#     ca._update_elem_params(ca_elem_params[i])
#     ca._update_fit_params(ca_fit_params[i])
#     absd_samples.append(ca.absd)
#
# absd_samples = np.array(absd_samples)
# cov_absd = np.cov(absd_samples, rowvar=False)
#
#
# print("alphas")
# print(ca_fit_params.T[-1])
#
# chollist = []
# for s in range(num_samples):
#     chollist.append(choLL(absd_samples[s], cov_absd))
#
# fig, ax = plt.subplots()
# ax.scatter(ca_fit_params.T[-1], chollist, label="cholli")
# # ax.set_xlim(-1e-8, 1e-8)
# # ax.set_ylim(0, 1e3)
# plt.savefig("plots/evenwiggliercholliplot.pdf")
#
#

###############################################################################

# mean_absd = np.average(absd_samples, axis=0)
#
# cov_absd_samples = np.array([[[
#     (absd_samples[s, a] - mean_absd[a]) * (absd_samples[s, b] - mean_absd[b])
#     for a in range(ca.nisotopepairs)] for b in range(ca.nisotopepairs)]
#     for s in range(num_samples)])  # not positive definite most of the time
#
#
# print("cov_absd_samples[0]")
# print(cov_absd_samples[0])
#
# print("alphas")
# print(ca_fit_params.T[-1])
#
# chollist = []
# for s in range(num_samples):
#     chollist.append(choLL(absd_samples[s], cov_absd_samples[s]))
#
# fig, ax = plt.subplots()
# ax.scatter(ca_fit_params.T[-1], chollist, label="cholli")
# ax.set_xlim(-1e-8, 1e-8)
# ax.set_ylim(0, 1e3)
