import numpy as np

from kifit.loadelems import Elem
from kifit.performfit import perform_odr, perform_linreg, linfit

import matplotlib.pyplot as plt


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


def draw_linfit():

    ca = Elem.get('Ca_testdata')

    betas_odr, sig_betas_odr, kperp1s, ph1s, sig_kperp1s, sig_ph1s = perform_odr(
        ca.mu_norm_isotope_shifts_in,  # [:, [0, 1]],
        ca.sig_mu_norm_isotope_shifts_in,  # [:, [0, 1]],
        reftrans_index=0)

    (betas_linreg, sig_betas_linreg, kperp1s_linreg, ph1s_linreg,
        sig_kperp1s_linreg, sig_ph1s_linreg) = perform_linreg(
        ca.mu_norm_isotope_shifts_in, reftrans_index=0)

    xvals = ca.mu_norm_isotope_shifts_in.T[0]
    sxvals = ca.sig_mu_norm_isotope_shifts_in.T[0]
    yvals = ca.mu_norm_isotope_shifts_in[:, 1:]
    syvals = ca.sig_mu_norm_isotope_shifts_in.T[1:]
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

# def test_mc():
#
#
# absd_init = ca.absd
# absd_list = []
# for i in range(num_samples):
#     if i % (num_samples // 100) == 0:
#         prog = np.round(i / num_samples * 100, 1)
#         print("Progress", prog, "%")
#     ca._update_elem_params(ca_elem_params[i])
#     ca._update_fit_params(ca_fit_params[i])
#     absd_list.append(ca.absd)
#
# absd_list = np.array(absd_list)
# cov_absd_np = np.cov(absd_list, rowvar=False)
# cov_absd = num_samples / (num_samples - 1) * np.array([[np.average([
#     (absd_list[s, a] - np.average(absd_list[:, a]))
#     * (absd_list[s, b] - np.average(absd_list[:, b]))
#     for s in range(num_samples)])
#     for a in range(ca.nisotopepairs)] for b in range(ca.nisotopepairs)])
#
# print("diff", cov_absd_np / cov_absd)
#
# print("cov")
# print(cov_absd)
#
# print("means", np.average(absd_list, axis=0))
# print("init", absd_init)
#
# print("absd list", absd_list)
#
#

if __name__ == "__main__":
    test_linfit()
