import numpy as np

from kifit.loadelems import Elem
from kifit.performfit import perform_odr, perform_linreg, linfit

import matplotlib.pyplot as plt


def test_linfit():

    ca = Elem.get('Ca_testdata')

    (betas_odr, sig_betas_odr, kperp1s_odr, ph1s_odr,
        sig_kperp1s_odr, sig_ph1s_odr) = perform_odr(
        ca.mu_norm_isotope_shifts, ca.sig_mu_norm_isotope_shifts,
        reftrans_index=0)

    (betas_linreg, sig_betas_linreg, kperp1s_linreg, ph1s_linreg,
        sig_kperp1s_linreg, sig_ph1s_linreg) = perform_linreg(
        ca.mu_norm_isotope_shifts, reftrans_index=0)

    assert betas_odr.shape == (ca.n_ntransitions - 1, 2)
    assert betas_linreg.shape == (ca.n_ntransitions - 1, 2)

    assert np.all(np.isclose(betas_odr, betas_linreg, rtol=1e-2))
    assert np.all(np.isclose(sig_betas_odr, sig_betas_linreg, rtol=1))
    assert np.all(np.isclose(kperp1s_odr, kperp1s_linreg, rtol=1e-2))
    assert np.all(np.isclose(ph1s_odr, ph1s_linreg, rtol=1e-2))
    assert np.all(np.isclose(sig_kperp1s_odr, sig_kperp1s_linreg, rtol=1))
    assert np.all(np.isclose(sig_ph1s_odr, sig_ph1s_linreg, rtol=1))

    xvals = ca.mu_norm_isotope_shifts.T[0]
    yvals = ca.mu_norm_isotope_shifts[:, 1:]

    betas_dat = np.array([np.polyfit(xvals, yvals[:, i], 1) for i in
        range(yvals.shape[1])])

    assert betas_dat.shape == (ca.n_ntransitions -1, 2)
    assert np.all(np.isclose(betas_dat, betas_odr, rtol=1e-2))


def draw_linfit():

    ca = Elem.get('Ca_testdata')

    betas_odr, sig_betas_odr, kperp1s, ph1s, sig_kperp1s, sig_ph1s = perform_odr(
        ca.mu_norm_isotope_shifts,  # [:, [0, 1]],
        ca.sig_mu_norm_isotope_shifts,  # [:, [0, 1]],
        reftrans_index=0)

    (betas_linreg, sig_betas_linreg, kperp1s_linreg, ph1s_linreg,
        sig_kperp1s_linreg, sig_ph1s_linreg) = perform_linreg(
        ca.mu_norm_isotope_shifts, reftrans_index=0)

    xvals = ca.mu_norm_isotope_shifts.T[0]
    sxvals = ca.sig_mu_norm_isotope_shifts.T[0]
    yvals = ca.mu_norm_isotope_shifts[:, 1:]
    syvals = ca.sig_mu_norm_isotope_shifts.T[1:]
    xfit = np.linspace(min(xvals) * 0.95, 1.05 * max(xvals), 1000)

    fig, ax = plt.subplots()
    transtyle = ['-', '--', ':']

    for i in range(yvals.shape[1]):
        yfit_odr = linfit(betas_odr[i], xfit)
        ax.plot(xfit, yfit_odr, 'orange', label='odr', linestyle=transtyle[i])
        yfit_linreg = linfit(betas_linreg[i], xfit)
        ax.plot(xfit, yfit_linreg, 'r', label='linreg', linestyle=transtyle[i])
        ax.scatter(xvals, yvals.T[i], color='b')
        ax.errorbar(xvals, yvals.T[i], xerr=sxvals, yerr=syvals.T[i], marker='o', ms=4)

    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_linfit()
