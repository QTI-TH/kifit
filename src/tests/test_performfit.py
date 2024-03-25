import numpy as np

from kifit.loadelems import Elem
from kifit.performfit import (perform_odr, perform_linreg,
    sample_alphaNP_fit_fixed_elemparams, sample_alphaNP_fit, sample_alphaNP_det)

# import matplotlib.pyplot as plt


def test_linfit():

    ca = Elem.get('Ca_testdata')

    (betas_odr, sig_betas_odr, kperp1s_odr, ph1s_odr,
        sig_kperp1s_odr, sig_ph1s_odr) = perform_odr(
        ca.mu_norm_isotope_shifts_in, ca.sig_mu_norm_isotope_shifts_in,
        reftrans_index=0)

    (betas_linreg, sig_betas_linreg, kperp1s_linreg, ph1s_linreg,
        sig_kperp1s_linreg, sig_ph1s_linreg) = perform_linreg(
        ca.mu_norm_isotope_shifts_in, reftrans_index=0)

    assert betas_odr.shape == (ca.ntransitions - 1, 2)
    assert betas_linreg.shape == (ca.ntransitions - 1, 2)

    assert np.all(np.isclose(betas_odr, betas_linreg, rtol=1e-2))
    assert np.all(np.isclose(sig_betas_odr, sig_betas_linreg, rtol=1))
    assert np.all(np.isclose(kperp1s_odr, kperp1s_linreg, rtol=1e-2))
    assert np.all(np.isclose(ph1s_odr, ph1s_linreg, rtol=1e-2))
    assert np.all(np.isclose(sig_kperp1s_odr, sig_kperp1s_linreg, rtol=1))
    assert np.all(np.isclose(sig_ph1s_odr, sig_ph1s_linreg, rtol=1))

    xvals = ca.mu_norm_isotope_shifts_in.T[0]
    yvals = ca.mu_norm_isotope_shifts_in[:, 1:]

    betas_dat = np.array([np.polyfit(xvals, yvals[:, i], 1) for i in
        range(yvals.shape[1])])

    assert betas_dat.shape == (ca.ntransitions -1, 2)
    assert np.all(np.isclose(betas_dat, betas_odr, rtol=1e-2))


def test_sample_alphaNP_det():

    ca = Elem.get('Ca_testdata')

    alphas_GKP_x0, sigalphas_GKP_x0 = sample_alphaNP_det(ca, 3, 100)
    alphas_NMGKP_x0, sigalphas_NMGKP_x0 = sample_alphaNP_det(ca, 3, 100, gkp=False)

    assert np.isclose(alphas_GKP_x0[0, 0], ca.alphaNP_GKP(), rtol=1e-50)
    assert np.isclose(alphas_NMGKP_x0[0, 0], ca.alphaNP_NMGKP(), rtol=1e-50)

    ca._update_Xcoeffs(1)

    alphas_GKP_x1, sigalphas_GKP_x1 = sample_alphaNP_det(ca, 3, 100)
    alphas_NMGKP_x1, sigalphas_NMGKP_x1 = sample_alphaNP_det(ca, 3, 100, gkp=False)

    assert np.isclose(alphas_GKP_x1[0, 0], ca.alphaNP_GKP(), rtol=1e-50)
    assert np.isclose(alphas_NMGKP_x1[0, 0], ca.alphaNP_NMGKP(), rtol=1e-50)

    alphas_GKP_mphivar, sigalphas_GKP_mphivar = sample_alphaNP_det(ca, 3, 100,
        mphivar=True)
    alphas_NMGKP_mphivar, sigalphas_NMGKP_mphivar = sample_alphaNP_det(ca, 3, 100,
        mphivar=True, gkp=False)

    assert np.isclose(alphas_GKP_x0[0, 0], alphas_GKP_mphivar[0, 0], rtol=1e-50)
    assert np.isclose(alphas_GKP_x0[0, 0], alphas_GKP_mphivar[0, 0], rtol=1e-50)

    assert np.isclose(alphas_GKP_x1[0, 0], alphas_GKP_mphivar[1, 0], rtol=1e-50)
    assert np.isclose(alphas_GKP_x1[0, 0], alphas_GKP_mphivar[1, 0], rtol=1e-50)


def test_sample_alphaNP_fit():

    ca = Elem.get('Ca_testdata')

    alpha_elemfixed, ll_elemfixed = sample_alphaNP_fit_fixed_elemparams(ca, 100)
    mean_alpha_elemfixed = np.average(alpha_elemfixed)
    sig_alpha_elemfixed = np.std(alpha_elemfixed)

    alpha_elemvar, ll_elemvar, elemparams = sample_alphaNP_fit(ca, 100)
    mean_alpha_elemvar = np.average(alpha_elemvar)
    sig_alpha_elemvar = np.std(alpha_elemvar)

    assert np.abs(
        mean_alpha_elemvar - mean_alpha_elemfixed) < sig_alpha_elemfixed
    assert np.abs(
        mean_alpha_elemvar - mean_alpha_elemfixed) < sig_alpha_elemvar


if __name__ == "__main__":
    test_linfit()
    test_sample_alphaNP_det()
    test_sample_alphaNP_fit()
