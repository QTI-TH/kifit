import numpy as np

from kifit.loadelems import Elem
from kifit.performfit_new import (perform_odr, perform_linreg,
    sample_alphaNP_det, determine_search_interval)

from test_fcts import iterative_mc_search

# import matplotlib.pyplot as plt


def test_linfit():

    ca = Elem('Ca_testdata')

    (betas_odr, sig_betas_odr, kperp1s_odr, ph1s_odr,
        sig_kperp1s_odr, sig_ph1s_odr) = perform_odr(
        ca.mu_norm_isotope_shifts_in, ca.sig_mu_norm_isotope_shifts_in,
        reftrans_index=0)

    (betas_linreg, sig_betas_linreg, kperp1s_linreg, ph1s_linreg,
        sig_kperp1s_linreg, sig_ph1s_linreg) = perform_linreg(
        ca.mu_norm_isotope_shifts_in, reftrans_index=0)

    assert betas_odr.shape == (ca.ntransitions - 1, 2)
    assert betas_linreg.shape == (ca.ntransitions - 1, 2)

    assert np.all(np.isclose(betas_odr, betas_linreg, atol=0, rtol=1e-2))
    assert np.all(np.isclose(sig_betas_odr, sig_betas_linreg, atol=0, rtol=1))
    assert np.all(np.isclose(kperp1s_odr, kperp1s_linreg, atol=0, rtol=1e-2))
    assert np.all(np.isclose(ph1s_odr, ph1s_linreg, atol=0, rtol=1e-2))
    assert np.all(np.isclose(sig_kperp1s_odr, sig_kperp1s_linreg, atol=0, rtol=1))
    assert np.all(np.isclose(sig_ph1s_odr, sig_ph1s_linreg, atol=0, rtol=1))

    xvals = ca.mu_norm_isotope_shifts_in.T[0]
    yvals = ca.mu_norm_isotope_shifts_in[:, 1:]

    betas_dat = np.array([np.polyfit(xvals, yvals[:, i], 1) for i in
        range(yvals.shape[1])])

    assert betas_dat.shape == (ca.ntransitions -1, 2)
    assert np.all(np.isclose(betas_dat, betas_odr, atol=0, rtol=1e-2))


def test_sample_alphaNP_det():

    ca = Elem('Ca_testdata')

    alphas_GKP_x0, sigalphas_GKP_x0 = sample_alphaNP_det(ca, 3, 100)
    alphas_NMGKP_x0, sigalphas_NMGKP_x0 = sample_alphaNP_det(ca, 3, 100, gkp=False)

    assert np.isclose(alphas_GKP_x0[0, 0], ca.alphaNP_GKP(), atol=0, rtol=1e-1)
    assert np.isclose(alphas_NMGKP_x0[0, 0], ca.alphaNP_NMGKP(),
        atol=0, rtol=1)

    ca._update_Xcoeffs(1)

    alphas_GKP_x1, sigalphas_GKP_x1 = sample_alphaNP_det(ca, 3, 100)
    alphas_NMGKP_x1, sigalphas_NMGKP_x1 = sample_alphaNP_det(ca, 3, 100, gkp=False)

    assert np.isclose(alphas_GKP_x1[0, 0], ca.alphaNP_GKP(), atol=0, rtol=100)
    assert np.isclose(alphas_NMGKP_x1[0, 0], ca.alphaNP_NMGKP(),
        atol=0, rtol=10)

    alphas_GKP_mphivar, sigalphas_GKP_mphivar = sample_alphaNP_det(ca, 3, 100,
        mphivar=True)
    alphas_NMGKP_mphivar, sigalphas_NMGKP_mphivar = sample_alphaNP_det(ca, 3, 100,
        mphivar=True, gkp=False)

    assert np.isclose(alphas_GKP_x0[0, 0], alphas_GKP_mphivar[0, 0],
        atol=0, rtol=1e-1)
    assert np.isclose(alphas_GKP_x0[0, 0], alphas_GKP_mphivar[0, 0],
        atol=0, rtol=1e-1)

    assert np.isclose(alphas_GKP_x1[0, 0], alphas_GKP_mphivar[1, 0],
        atol=0, rtol=1e-1)
    assert np.isclose(alphas_GKP_x1[0, 0], alphas_GKP_mphivar[1, 0],
        atol=0, rtol=1e-1)


# def sample_alphaNP_fit_fixed_elemparams(elem, nsamples, mphivar=False):
#     """
#     Keeping the element parameters fixed to their mean values, generate nsamples
#     of
#
#        alphaNP ~ N(0, sig[alphaNP_init]).
#
#     If mphivar=True, this procedure is repeated for all X-coefficients provided
#     for elem.
#
#     Returns two (nmphi, nsamples)-dimensional numpy arrays: one with the values
#     of alphaNP, the other with the respective loglikelihoods.
#
#     """
#     print(
#         """Generating %s samples for the orthogonal distance King fit with
#     fixed element parameters. mphi is %s varied."""
#         % (nsamples, ("" if mphivar else "not"))
#     )
#
#     fitparams = elem.means_fit_params
#     fitparamsamples = np.tensordot(np.ones(nsamples), fitparams, axes=0)
#
#     if mphivar:
#         Nx = len(elem.Xcoeff_data)
#     else:
#         Nx = 1
#
#     alphalist = []
#     llist = []
#
#     for x in range(Nx):
#         elem._update_Xcoeffs(x)
#         sigalphaNP = elem.sig_alphaNP_init
#         alphaNPsamples = np.random.normal(fitparams[-1], sigalphaNP, nsamples)
#         alphalist.append(alphaNPsamples)
#         fitparamsamples[:, -1] = alphaNPsamples
#
#         absdsamples = []
#         for s in range(nsamples):
#             elem._update_fit_params(fitparamsamples[s])
#             absdsamples.append(elem.absd)
#         llist.append(get_llist(np.array(absdsamples), nsamples))
#
#     alphalist = elem.dnorm * np.array(alphalist)
#     llist = elem.dnorm * np.array(llist)
#
#     return alphalist, llist
#
#
# def test_sample_alphaNP_fit():
#
#     ca = Elem.get('Ca_testdata')
#
#     alpha_elemfixed, ll_elemfixed = sample_alphaNP_fit_fixed_elemparams(ca, 100)
#     mean_alpha_elemfixed = np.average(alpha_elemfixed)
#     sig_alpha_elemfixed = np.std(alpha_elemfixed)
#
#     alpha_elemvar, ll_elemvar, elemparams = sample_alphaNP_fit(ca, 100)
#     mean_alpha_elemvar = np.average(alpha_elemvar)
#     sig_alpha_elemvar = np.std(alpha_elemvar)
#
#     assert np.abs(
#         mean_alpha_elemvar - mean_alpha_elemfixed) < sig_alpha_elemfixed
#     assert np.abs(
#         mean_alpha_elemvar - mean_alpha_elemfixed) < sig_alpha_elemvar


def test_logL_minimisation():

    np.random.seed(1)

    ca = Elem('Ybmin')
    x0 = 0
    ca._update_Xcoeffs(x0)

    res_it = iterative_mc_search(
        elem=ca,
        nelemsamples_search=100,
        nalphasamples_search=100,
        nexps=10,
        nelemsamples_exp=100,
        nalphasamples_exp=100,
        nsigmas=2,
        nblocks=1,
        scalefactor=.3,
        maxiter=20,
        sigalphainit=1e-7,
        plot_output=False,
        xind=x0,
        mphivar=False)
# [
#         np.array(alphas_exps), np.array(delchisqs_exps),
#         delchisqcrit,
#         best_alpha_pts, sig_alpha_pts,
#         LB, sig_LB, UB, sig_UB,
#         xind]

    best_alpha_it = np.median(res_it[3])
    sig_best_alpha_it = np.max(res_it[4])

    best_alpha_scp, sig_best_alpha_scp = determine_search_interval(
        ca,
        nsearches=10,
        nelemsamples_search=100,
        alpha0=0,
        maxiter=100)

    assert np.isclose(best_alpha_it, best_alpha_scp, atol=0, rtol=1e-3)

    min_it = best_alpha_it - sig_best_alpha_it
    max_it = best_alpha_it + sig_best_alpha_it

    min_scp = best_alpha_scp - sig_best_alpha_scp
    max_scp = best_alpha_scp + sig_best_alpha_scp

    assert ((min_it <= max_scp) & (min_scp <= max_it))


if __name__ == "__main__":
    test_linfit()
    test_sample_alphaNP_det()
    # test_sample_alphaNP_fit()
    test_logL_minimisation()
