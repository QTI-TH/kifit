from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


from qiss.loadelems import Elem
from qiss.kingmc import perform_odr, perform_linreg, linfit


ca = Elem.get('Ca_testdata')

betas_odr, sig_betas_odr, kperp1s, ph1s, sig_kperp1s, sig_ph1s = perform_odr(
    ca.mu_norm_isotope_shifts,  # [:, [0, 1]],
    ca.sig_mu_norm_isotope_shifts,  # [:, [0, 1]],
    reftrans_index=0)

print("kperp1s", kperp1s)
# ca.update_fit_params([kperp1s, ph1s, 0])



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




