from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from kifit.loadelems import Elem
from kifit.performfit import perform_odr, perform_linreg, linfit

from scipy.stats import multivariate_normal
np.random.seed(1)

ca = Elem.get('Ca_testdata')

print("ca ma in")
print(ca.m_a_in)
print("ca ma")
print(ca.m_a)
print("ca nu in")
print(ca.nu_in)
print("ca nu")
print(ca.nu)

print("ca nu flat")
print(ca.nu.flatten())

print("means fit params")
print(ca.means_fit_params)

print("means input params")
print(ca.means_input_params)


num_samples = 5000

ca_elem_params = multivariate_normal.rvs(
    ca.means_input_params,
    ca.stdevs_input_params,
    size=num_samples
)

ca_fit_params = multivariate_normal.rvs(
    ca.means_fit_params,
    ca.stdevs_fit_params,
    size=num_samples
)

absd_init = ca.absd
absd_list = []
for i in range(num_samples):
    if i % (num_samples // 100) == 0:
        prog = np.round(i / num_samples * 100, 1)
        print("Progress", prog, "%")
    ca._update_elem_params(ca_elem_params[i])
    ca._update_fit_params(ca_fit_params[i])
    absd_list.append(ca.absd)

absd_list = np.array(absd_list)

cov_absd_np = np.cov(absd_list.T)
print("cov_absd_np", cov_absd_np)
covmat = np.array([[np.cov(absd_list.T[a], absd_list.T[b])
    for b in range(ca.nisotopepairs)] for a in range(ca.nisotopepairs)])
print("covmat", covmat)

cov_absd = np.array([[np.average([
    (absd_list[s, a] - absd_init[a]) * (absd_list[s, b] - absd_init[b])
    for s in range(num_samples)])
    for a in range(ca.nisotopepairs)] for b in range(ca.nisotopepairs)])
print("cov")
print(cov_absd)

print("absd list", absd_list)


#
# # get element and linear fit parameters
# elem = Elem.get('Ca_testdata')
#
# betas_odr, sig_betas_odr, kperp1s, ph1s, sig_kperp1s, sig_ph1s = perform_odr(
#     elem.mu_norm_isotope_shifts,  # [:, [0, 1]],
#     elem.sig_mu_norm_isotope_shifts,  # [:, [0, 1]],
#     reftrans_index=0)
#
# (betas_linreg, sig_betas_linreg, kperp1s_linreg, ph1s_linreg,
#     sig_kperp1s_linreg, sig_ph1s_linreg) = perform_linreg(
#     elem.mu_norm_isotope_shifts, reftrans_index=0)
#
# xvals = elem.mu_norm_isotope_shifts.T[0]
# sxvals = elem.sig_mu_norm_isotope_shifts.T[0]
# yvals = elem.mu_norm_isotope_shifts[:, 1:]
# syvals = elem.sig_mu_norm_isotope_shifts.T[1:]
# xfit = np.linspace(min(xvals) * 0.95, 1.05 * max(xvals), 1000)
#
# fig, ax = plt.subplots()
# transtyle = ['-', '--', ':']
#
# for i in range(yvals.shape[1]):
#     yfit_odr = linfit(betas_odr[i], xfit)
#     ax.plot(xfit, yfit_odr, 'orange', label='odr', linestyle=transtyle[i])
#     yfit_linreg = linfit(betas_linreg[i], xfit)
#     ax.plot(xfit, yfit_linreg, 'r', label='linreg', linestyle=transtyle[i])
#     ax.scatter(xvals, yvals.T[i], color='b')
#     ax.errorbar(xvals, yvals.T[i], xerr=sxvals, yerr=syvals.T[i], marker='o', ms=4)
#
# plt.tight_layout()
# plt.legend()
# plt.show()
