import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from kifit.performfit import (
    get_confints,
    get_delchisq,
    linfit,
    perform_linreg,
    perform_odr,
)

_plot_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
)
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
        elem.mu_norm_isotope_shifts_in,
        elem.sig_mu_norm_isotope_shifts_in,
        reftrans_index=0,
    )

    (
        betas_linreg,
        sig_betas_linreg,
        kperp1s_linreg,
        ph1s_linreg,
        sig_kperp1s_linreg,
        sig_ph1s_linreg,
    ) = perform_linreg(elem.mu_norm_isotope_shifts_in, reftrans_index=0)

    xvals = elem.mu_norm_isotope_shifts_in.T[0]
    sxvals = elem.sig_mu_norm_isotope_shifts_in.T[0]
    yvals = elem.mu_norm_isotope_shifts_in[:, 1:]
    syvals = elem.sig_mu_norm_isotope_shifts_in.T[1:]
    xfit = np.linspace(min(xvals) * 0.95, 1.05 * max(xvals), 1000)

    fig, ax = plt.subplots()
    transtyle = ["-", "--", ":", "-.", (0, (1, 10)), (5, (10, 3)), (0, (5, 10))]

    for i in range(yvals.shape[1]):
        yfit_odr = linfit(betas_odr[i], xfit)
        ax.plot(xfit, yfit_odr, "orange", label="odr", linestyle=transtyle[i])
        yfit_linreg = linfit(betas_linreg[i], xfit)
        ax.plot(xfit, yfit_linreg, "r", label="linreg", linestyle=transtyle[i])
        ax.scatter(xvals, yvals.T[i], color="b")
        ax.errorbar(xvals, yvals.T[i], xerr=sxvals, yerr=syvals.T[i], marker="o", ms=4)

    plt.tight_layout()
    plt.legend()
    plt.savefig(_plot_path + "/" + plotname + "_" + elem.id + ".pdf")
    if show:
        plt.show()


def draw_mc_output(
    elem,
    paramlist,
    llist,
    x=0,
    confints=True,
    nsigmas=[1, 2],
    dof=1,
    showGKP=False,
    showNMGKP=False,
    xlabel="x",
    ylabel=r"$\Delta \chi^2$",
    plotname="testplot",
    xlims=[None, None],
    ylims=[None, None],
    show=False,
):
    """
    Draw 2-dimensional scatter plot showing the likelihood associated with the
    parameter values given in paramlist. If the lists were computed for multiple
    X-coefficients, the argument x can be used to access a given set of samples.
    The resulting plot is saved in plots directory under plotname.

    """
    delchisqlist = get_delchisq(llist[x])

    fig, ax = plt.subplots()
    ax.scatter(paramlist, delchisqlist, s=1)

    if confints:
        for ns in nsigmas:
            delchisqcrit, parampos = get_confints(
                paramlist, delchisqlist, ns, dof, verbose=False
            )
            ax.axvspan(np.min(parampos), np.max(parampos), alpha=0.5, color="darkgreen")
            if ns == 1:
                hlinels = "--"
            else:
                hlinels = "-"
            ax.axhline(y=delchisqcrit, color="orange", linewidth=1, linestyle=hlinels)
    if showGKP:
        for aNP in elem.alphaNP_GKP:
            ax.axhline  # continue here

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    plt.savefig(_plot_path + "/" + plotname + ".pdf")
    if show:
        plt.show()
    return 0


def plot_loss_varying_alphaNP(
    alphaNP_list, ll_list, filename="", save=True, document_width=0.5
):
    """Scatterplot of ``alphaNP`` versus ``ll``."""

    best_index = np.argmin(ll_list)
    ll_list = get_delchisq(ll_list[0])

    plt.figure(figsize=(document_width * 10, document_width * 10 * 6 / 8))
    plt.scatter(alphaNP_list, ll_list, color="black", s=10, alpha=0.6)
    plt.scatter(
        alphaNP_list[best_index],
        ll_list[best_index],
        label="Best candidate",
        color="red",
        s=50,
        alpha=0.6,
    )
    plt.xlabel(r"$\alpha_{NP}$")
    plt.ylabel(r"$LL$")
    plt.legend()
    plt.grid(True)

    if save:
        plt.savefig(f"alpha_vs_ll_{filename}.pdf", bbox_inches="tight")
