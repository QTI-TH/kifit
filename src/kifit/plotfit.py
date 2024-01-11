import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

from kifit.performfit import (linfit, perform_linreg, perform_odr, get_delchisq,
    get_confints)

_plot_path = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'plots'
))
if not os.path.exists(_plot_path):
    os.makedirs(_plot_path)


def draw_linfit(elem, plotname="linfit", show=False):
    """
    Draw, King plot data and output of linear regression and orthogonal distance
    regression. Use to check linear fit.

    Input: instance of Elem class, plot name, boolean whether or not to show plot
    Output: plot saved in plots directory under plotname.

    """

    betas_odr, sig_betas_odr, kperp1s, ph1s, sig_kperp1s, sig_ph1s = perform_odr(
        elem.mu_norm_isotope_shifts_in, elem.sig_mu_norm_isotope_shifts_in,
        reftrans_index=0)

    (betas_linreg, sig_betas_linreg, kperp1s_linreg, ph1s_linreg,
        sig_kperp1s_linreg, sig_ph1s_linreg) = perform_linreg(
        elem.mu_norm_isotope_shifts_in, reftrans_index=0)

    xvals = elem.mu_norm_isotope_shifts_in.T[0]
    sxvals = elem.sig_mu_norm_isotope_shifts_in.T[0]
    yvals = elem.mu_norm_isotope_shifts_in[:, 1:]
    syvals = elem.sig_mu_norm_isotope_shifts_in.T[1:]
    xfit = np.linspace(min(xvals) * 0.95, 1.05 * max(xvals), 1000)

    fig, ax = plt.subplots()
    transtyle = ['-', '--', ':', '-.', (0, (1, 10)), (5, (10, 3)), (0, (5, 10))]

    for i in range(yvals.shape[1]):
        yfit_odr = linfit(betas_odr[i], xfit)
        ax.plot(xfit, yfit_odr, 'orange', label='odr', linestyle=transtyle[i])
        yfit_linreg = linfit(betas_linreg[i], xfit)
        ax.plot(xfit, yfit_linreg, 'r', label='linreg', linestyle=transtyle[i])
        ax.scatter(xvals, yvals.T[i], color='b')
        ax.errorbar(xvals, yvals.T[i], xerr=sxvals, yerr=syvals.T[i], marker='o', ms=4)

    plt.tight_layout()
    plt.legend()
    plt.savefig(_plot_path + "/" + plotname + "_" + elem.id + ".pdf")
    if show:
        plt.show()


def draw_mc_output(paramlist, llist,
        confints=True, nsigmas=[1, 2], dof=1,
        xlabel='x', ylabel=r"$\Delta \chi^2$", plotname='testplot',
        xlims=[None, None], ylims=[None, None], show=False):
    """
    Draw 2-dimensional scatter plot starting from xvals, yvals.
    (Useful to show output of Monte Carlo.) Plot is saved in plots directory
    under plotname.

    """
    delchisqlist = get_delchisq(llist)

    fig, ax = plt.subplots()
    ax.scatter(paramlist, delchisqlist, s=1)

    if confints:
        for ns in nsigmas:
            delchisqcrit, parampos = get_confints(
                paramlist, delchisqlist, ns, dof)
            ax.axvspan(np.min(parampos), np.max(parampos), alpha=.5,
            color='darkgreen')
            if ns==1:
                hlinels = '--'
            else:
                hlinels = '-'
            ax.axhline(y=delchisqcrit, color='orange', linewidth=1,
                linestyle=hlinels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    plt.savefig(_plot_path + "/" + plotname + ".pdf")
    if show:
        plt.show()
    return 0
