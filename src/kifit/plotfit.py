import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, interp1d, splrep
from scipy.optimize import curve_fit

from kifit.performfit import (
    get_all_alphaNP_bounds,
    get_confint,
    get_delchisq,
    get_delchisq_crit,
    get_minpos_maxneg_alphaNP_bounds,
    get_odr_residuals,
    # interpolate_mphi_alphaNP_fit,
    linfit,
    # linfit_x,
    parabola,
    perform_linreg,
    perform_odr,
    sample_alphaNP_det,
)

# import pandas as pd


_plot_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
)

if not os.path.exists(_plot_path):
    os.makedirs(_plot_path)

###############################################################################

default_colour = [
    "#ff7f0e",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

det_colour = "#1f77b4"
fit_colour = "#2ca02c"


###############################################################################


def plot_linfit(elem, magnifac=1, resmagnifac=1):
    """
    Plot, King plot data and output of linear regression and orthogonal distance
    regression. Use to check linear fit.

    Input: instance of Elem class, plot name, boolean whether or not to show plot
    Output: plot saved in plots directory with name linfit + elem.

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
    sigxvals = elem.sig_mu_norm_isotope_shifts_in.T[0]
    yvals = elem.mu_norm_isotope_shifts_in[:, 1:].T
    sigyvals = elem.sig_mu_norm_isotope_shifts_in[:, 1:].T

    AAp = np.array([elem.a_nisotope, elem.ap_nisotope]).T

    xmin = 0.95 * min(xvals)
    xmax = 1.05 * max(xvals)
    xfit = np.linspace(xmin, xmax, 1000)

    fig = plt.figure()
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 1), (2, 0))
    ax3 = plt.subplot2grid((4, 1), (3, 0))
    ax1.set_title("King Plot")

    ax2.axhline(y=0, c="k")
    ax3.axhline(y=0, c="k")

    for i in range(yvals.shape[0]):

        yfit_linreg = linfit(betas_linreg[i], xfit)
        residuals_linreg_i, sigresiduals_linreg_i = get_odr_residuals(
            betas_linreg[i], xvals, yvals[i], sigxvals, sigyvals[i]
        )

        yfit_odr_i = linfit(betas_odr[i], xfit)

        residuals_odr_i, sigresiduals_odr_i = get_odr_residuals(
            betas_odr[i], xvals, yvals[i], sigxvals, sigyvals[i]
        )

        ax1.plot(xfit, yfit_odr_i, color=default_colour[i], label="i=" + str(i + 2))

        ax1.plot(xfit, yfit_linreg, color=default_colour[i], linestyle="--")

        for a in range(yvals.shape[1]):
            ax1.annotate(
                "(" + str(int(AAp[a, 0])) + "," + str(int(AAp[a, 1])) + ")",
                (xvals[a], yvals[i, a]),
                fontsize=8,
            )

        ax1.errorbar(
            xvals,
            yvals[i],
            xerr=sigxvals,
            yerr=sigyvals[i],
            linestyle="none",
            color=default_colour[i],
            marker="o",
            ms=4,
        )

        ax2.errorbar(
            xvals,
            magnifac * residuals_odr_i,
            xerr=sigresiduals_odr_i,
            yerr=resmagnifac * magnifac * sigresiduals_odr_i,
            linestyle="none",
            color=default_colour[i],
            marker="o",
            ms=4,
            capsize=2,
        )

        ax3.errorbar(
            xvals,
            magnifac * residuals_odr_i,
            xerr=sigresiduals_odr_i,
            yerr=resmagnifac * magnifac * sigresiduals_odr_i,
            linestyle="none",
            color=default_colour[i],
            marker="o",
            ms=4,
            capsize=2,
        )

    ax1.set_xlim(xmin, xmax)
    ax1.set_xlabel(r"$\tilde{\nu}_1~$[Hz]")
    ax1.set_ylabel(r"$\tilde{\nu}_i~$[Hz]")
    ax1.legend(fontsize="6")
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylabel(r"ODR Resid.")
    ax3.set_xlim(xmin, xmax)
    ax3.set_ylabel(r"Lin. Reg. Resid.")

    plt.tight_layout()
    plt.savefig(_plot_path + "/linfit_" + elem.id + ".pdf")

    return fig, ax1, ax2, ax3

# leaving this here for reference could rename draw_final_mc_output ?
def plot_alphaNP_ll(elem, alphalist, llist, x=0, nsigmas=2, dof=1,
    gkpdims=[], nmgkpdims=[], parabolaparams=[],
    plotname="", xlabel=r"$\alpha_{\mathrm{NP}}$", ylabel=r"$\Delta \chi^2$",
    xlims=[None, None], ylims=[None, None], show=False,
):
    """
    Plot 2-dimensional scatter plot showing the likelihood associated with the
    parameter values given in alphalist. If the lists were computed for multiple
    X-coefficients, the argument x can be used to access a given set of samples.
    The resulting plot is saved in plots directory under alphaNP_ll + elem.

    """

    delchisqlist_x = get_delchisq(llist[x])

    fig, ax = plt.subplots()
    ax.scatter(alphalist[x], delchisqlist_x, s=1, c="b")

    if len(parabolaparams) == 3:
        delchisqcrit_x = get_delchisq_crit(nsigmas=nsigmas)
        parampos_x = get_confint(min(alphalist[x]), max(alphalist[x]),
            parabolaparams, nsigmas)
        ax.axvspan(
            np.min(parampos_x), np.max(parampos_x), alpha=0.5, color=fit_colour
        )
        print("delchisqcrit", delchisqcrit_x)
        print("parampos", np.min(parampos_x))
        ax.axhline(y=delchisqcrit_x, color="orange", lw=1, ls="--")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])

    if plotname == "":
        prettyplotname = ""
    else:
        prettyplotname = plotname + "_"

    plt.savefig(
        _plot_path
        + "/alphaNP_ll_"
        + elem.id
        + "_"
        + prettyplotname
        + "x"
        + str(x)
        + "_"
        + str(len(alphalist[0]))
        + "_fit_samples.pdf"
    )
    return fig, ax


def plot_mphi_alphaNP_det_bound(
    ax1,
    ax2,
    elem,
    dimindex,
    dim,
    nsamples,
    nsigmas,
    minpos,
    maxneg,
    gkp=True,
    showbestdetbounds=False,
    showalldetbounds=False,
):
    """
    Plot GKP/NMGKP bounds for one dimension dim.

    """
    if gkp:
        method_tag = "GKP"

    else:
        method_tag = "NMGKP"

    alphas, sigalphas = sample_alphaNP_det(elem, dim, nsamples, mphivar=True, gkp=gkp)

    if showalldetbounds:
        alphaNP_UBs, alphaNP_LBs = get_all_alphaNP_bounds(
            alphas, sigalphas, nsigmas=nsigmas
        )

        for p in range(alphas.shape[1]):
            if p == 0:
                scatterlabel = elem.id + ", dim " + str(dim) + " " + method_tag
            else:
                scatterlabel = None

            ax1.scatter(
                elem.mphis,
                (alphaNP_UBs.T)[p],
                s=0.5,
                color=default_colour[dimindex],
                alpha=0.3,
                label=scatterlabel,
            )
            ax2.scatter(
                elem.mphis,
                (alphaNP_UBs.T)[p],
                s=0.5,
                color=default_colour[dimindex],
                alpha=0.3,
                label=scatterlabel,
            )

            ax1.scatter(
                elem.mphis,
                np.abs((alphaNP_LBs.T)[p]),
                s=0.5,
                color=default_colour[dimindex],
                alpha=0.3,
            )
            ax2.scatter(
                elem.mphis,
                (alphaNP_LBs.T)[p],
                s=0.5,
                color=default_colour[dimindex],
                alpha=0.3,
            )

    minpos_alphas, maxneg_alphas = get_minpos_maxneg_alphaNP_bounds(
        alphas, sigalphas, nsigmas
    )

    if showbestdetbounds:
        ax1.scatter(
            elem.mphis,
            np.nanmax(minpos_alphas, -maxneg_alphas, axis=1),
            s=6,
            color=default_colour[dimindex],
            label=elem.id + ", dim " + str(dim) + " " + method_tag + "_best",
        )
        ax2.scatter(
            elem.mphis,
            minpos_alphas,
            s=6,
            color=default_colour[dimindex],
            label=elem.id + ", dim " + str(dim) + " " + method_tag + "_best",
        )
        ax2.scatter(elem.mphis, maxneg_alphas, s=6, color=default_colour[dimindex])

    if dimindex == 0:
        minpos = minpos_alphas
        maxneg = maxneg_alphas

    else:
        minpos = np.fmin(minpos, minpos_alphas)
        maxneg = np.fmax(maxneg, maxneg_alphas)

    return ax1, ax2, minpos, maxneg


def plot_mphi_alphaNP_fit_bound(ax1, ax2, elem, bestalphas, sigbestalphas,
    lowerbounds, siglowerbounds, upperbounds, sigupperbounds,
        plotabs=True, showallowedfitpts=False):

    ax1.scatter(elem.mphis, bestalphas, color='k', marker="*")
    ax1.errorbar(elem.mphis, bestalphas, yerr=sigbestalphas, ecolor='k')
    ax1.plot(elem.mphis, lowerbounds, color=fit_colour)
    ax1.errorbar(elem.mphis, lowerbounds, yerr=siglowerbounds, ecolor=fit_colour)
    ax1.plot(elem.mphis, upperbounds, color=fit_colour)
    ax1.errorbar(elem.mphis, upperbounds, yerr=sigupperbounds, ecolor=fit_colour)

    return ax1


def set_axes(
    ax1,
    ax2,
    xlims,
    ylims,
    linthreshold,
    elem,
    minpos,
    maxneg,
    ax1_fitinterpolpts,
    ax2max_fitinterpolpts,
    ax2min_fitinterpolpts,
):

    if xlims[0] is not None:
        ax1.set_xlim(left=xlims[0])
        ax2.set_xlim(left=xlims[0])

    else:
        ax1.set_xlim(left=min(elem.mphis))
        ax2.set_xlim(left=min(elem.mphis))

    if xlims[1] is not None:
        ax1.set_xlim(right=xlims[1])
        ax2.set_xlim(right=xlims[1])

    else:
        ax1.set_xlim(right=max(elem.mphis))
        ax2.set_xlim(right=max(elem.mphis))

    if ylims[0] is not None:
        ymin_ax1 = ylims[0]
        ymin_ax2 = ylims[0]
    else:
        ymin_ax1 = (
            np.nanmin([np.abs(maxneg), np.abs(minpos), np.abs(ax1_fitinterpolpts)]) / 10
        )
        ymin_ax2 = np.nanmin([np.nanmin(maxneg), np.nanmin(ax2min_fitinterpolpts)])

    ax1.set_ylim(bottom=ymin_ax1)
    ax2.set_ylim(bottom=ymin_ax2)

    if ylims[1] is not None:
        ymax = ylims[1]
    else:
        ymax = np.nanmax([np.nanmax(minpos), np.nanmax(ax1_fitinterpolpts)])
    ax1.set_ylim(top=ymax)
    ax2.set_ylim(top=ymax)

    ax1.set_xlabel(r"$m_\phi~$[eV]")
    ax1.set_xlabel(r"$m_\phi~$[eV]")
    ax1.set_ylabel(r"$|\alpha_{\mathrm{NP}}/\alpha_{\mathrm{EM}}|$")
    ax2.set_ylabel(r"$\alpha_{\mathrm{NP}}/\alpha_{\mathrm{EM}}$")

    ax2.axhline(y=0, c="k")

    # x-axis
    ax1.set_xscale("log")
    ax2.set_xscale("log")

    # y-axis
    ax1.set_yscale("log")

    if linthreshold is None:
        linlim = 10 ** (
            np.floor(
                np.log10(
                    np.nanmin(
                        [
                            np.nanmin(np.abs(minpos)),
                            np.nanmin(np.abs(maxneg)),
                            np.nanmin(np.abs(ax2max_fitinterpolpts)),
                            np.nanmin(np.abs(ax2min_fitinterpolpts)),
                        ]
                    )
                )
                - 1
            )
        )
        print("linthreshold", linlim)
    else:
        linlim = linthreshold
    ax2.set_yscale("symlog", linthresh=linlim)

    return ax1, ax2, ymin_ax2, ymax


def plot_mphi_alphaNP(
    elem,
    alphalist,
    llist,
    gkpdims,
    nmgkpdims,
    ndetsamples,
    nsigmas=2,
    plotabs=True,
    xlims=[None, None],
    ylims=[None, None],
    linthreshold=None,
    plotname="",
    showallowedfitpts=False,
    showbestdetbounds=False,
    showalldetbounds=False,
):
    """
    Plot the most stringent nsigmas-bounds on both positive and negative
    alphaNP, derived using the Generalised King-plot formula of dimensions d
    listed in dims and save the output under plotname in the plots directory.
    If showall=True, all bounds GKP bounds of the appropriate dimensions are
    shown.

    """
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    # fit
    ###########################################################################
    if len(alphalist) > 0:
        (ax1, ax2, ax1_fitinterpolpts, ax2max_fitinterpolpts, ax2min_fitinterpolpts) = (
            plot_mphi_alphaNP_fit_bound(
                ax1,
                ax2,
                elem,
                alphalist,
                llist,
                nsigmas=nsigmas,
                showallowedfitpts=showallowedfitpts,
            )
        )
    else:
        ax1_fitinterpolpts = np.array([np.nan] * len(elem.mphis))
        ax2max_fitinterpolpts = np.array([np.nan] * len(elem.mphis))
        ax2min_fitinterpolpts = np.array([np.nan] * len(elem.mphis))

    # determinant methods
    ###########################################################################

    minpos = np.array([])
    maxneg = np.array([])

    for d, dim in enumerate(gkpdims):
        ax1, ax2, minpos, maxneg = plot_mphi_alphaNP_det_bound(
            ax1,
            ax2,
            elem,
            d,
            dim,
            ndetsamples,
            nsigmas,
            minpos,
            maxneg,
            gkp=True,
            showbestdetbounds=showbestdetbounds,
            showalldetbounds=showalldetbounds,
        )

    for d, dim in enumerate(nmgkpdims):
        ax1, ax2, minpos, maxneg = plot_mphi_alphaNP_det_bound(
            ax1,
            ax2,
            elem,
            d + len(gkpdims),
            dim,
            ndetsamples,
            nsigmas,
            minpos,
            maxneg,
            gkp=False,
            showbestdetbounds=showbestdetbounds,
            showalldetbounds=showalldetbounds,
        )

    # formatting + plotting combined det bound
    ###########################################################################
    gkp_label = (
        ("(" + ", ".join(str(gd) for gd in gkpdims) + ")-dim GKP")
        if len(gkpdims) > 0
        else ""
    )
    nmgkp_label = (
        ("(" + ", ".join(str(nmd) for nmd in nmgkpdims) + ")-dim NMGKP")
        if len(nmgkpdims) > 0
        else ""
    )
    label_coupling = " + " if (gkp_label != "" and nmgkp_label != "") else ""

    det_label = gkp_label + label_coupling + nmgkp_label

    ax1, ax2, ymin_ax2, ymax = set_axes(
        ax1,
        ax2,
        xlims,
        ylims,
        linthreshold,
        elem,
        minpos,
        maxneg,
        ax1_fitinterpolpts,
        ax2max_fitinterpolpts,
        ax2min_fitinterpolpts,
    )

    ax1.plot(
        elem.mphis,
        np.nanmin([np.abs(minpos), np.abs(maxneg)], axis=0),
        color=det_colour,
    )

    ax1.fill_between(
        elem.mphis,
        np.nanmin([np.abs(minpos), np.abs(maxneg)], axis=0),
        ymax,
        label=det_label + " " + str(nsigmas) + r"$\sigma$-excluded",
        color=det_colour,
        alpha=0.3,
    )

    ax2.plot(elem.mphis, minpos, color=det_colour)  # , label=det_label)
    ax2.plot(elem.mphis, maxneg, color=det_colour)
    ax2.scatter(elem.mphis, minpos, color=det_colour, s=1)
    ax2.scatter(elem.mphis, maxneg, color=det_colour, s=1)

    ax2.fill_between(
        elem.mphis,
        minpos,
        ymax,
        color=det_colour,
        alpha=0.3,
        label=det_label + " " + str(nsigmas) + r"$\sigma$-excluded",
    )
    ax2.fill_between(elem.mphis, ymin_ax2, maxneg, color=det_colour, alpha=0.3)

    if len(alphalist) > 0:
        ax1.fill_between(
            elem.mphis,
            ax1_fitinterpolpts,
            ymax,
            color=fit_colour,
            alpha=0.3,
            label="fit " + str(nsigmas) + r"$\sigma$-excluded",
        )
        ax2.fill_between(
            elem.mphis,
            ax2max_fitinterpolpts,
            ymax,
            color=fit_colour,
            alpha=0.3,
            label="fit " + str(nsigmas) + r"$\sigma$-excluded",
        )
        ax2.fill_between(
            elem.mphis,
            ymin_ax2,
            ax2min_fitinterpolpts,
            color=fit_colour,
            alpha=0.3,
            label="fit " + str(nsigmas) + r"$\sigma$-excluded",
        )

    ax1.legend(loc="upper left", fontsize="9")
    ax2.legend(loc="upper left", fontsize="9")

    if plotname == "":
        prettyplotname = ""
    else:
        prettyplotname = plotname + "_"

    fig1.savefig(
        _plot_path
        + "/mphi_abs_alphaNP_"
        + elem.id
        + "_"
        + prettyplotname
        + str(len(alphalist[0]))
        + "_fit_samples.pdf"
    )

    fig2.savefig(
        _plot_path
        + "/mphi_alphaNP_"
        + elem.id
        + "_"
        + prettyplotname
        + str(len(alphalist[0]))
        + "_fit_samples.pdf"
    )

    return fig1, ax1, fig2, ax2


def draw_mc_output(alphalist, delchisqlist, parabolaparams,
        nsigmas=2,
        xlabel=r"$\alpha_{\mathrm{NP}}$", ylabel=r"$\Delta \chi^2$",
        plotname="mc_output",
        xlims=[None, None], ylims=[None, None]):
    """
    Draw 2-dimensional scatter plot showing the likelihood associated with the
    parameter values given in alphalist, as well as the parabola defined by the
    parameters `parabolaparams`.
    The resulting plot is saved in plots directory under plotname.

    """
    fig, ax = plt.subplots()

    ax.scatter(alphalist, delchisqlist, s=1, alpha=0.5, color="royalblue")

    ll_fit = parabola(alphalist, *parabolaparams)
    best_alpha = alphalist[np.argmin(ll_fit)]

    ax.plot(alphalist, ll_fit,
        color="black", ls="--", lw=1,
        label=rf"$\alpha$ fit min: {best_alpha:.4e}")

    delchisqcrit = get_delchisq_crit(nsigmas=nsigmas)
    ax.axhline(y=delchisqcrit, color="orange", lw=1, ls="--")

    confint = get_confint(alphalist, delchisqlist, delchisqcrit)
    lb = confint[0]
    ub = confint[1]
    ax.axvspan(lb, ub, alpha=.5, color="darkgreen")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    plt.legend()
    plt.savefig(_plot_path + "/" + plotname + ".pdf")

    return ax


def draw_final_mc_output(alphas, delchisqs, parabolaparams, delchisqcrit,
        bestalphaparabola=None, sigbestalphaparabola=None,
        bestalphapt=None, sigbestalphapt=None,
        lb=None, siglb=None, ub=None, sigub=None,
        nsigmas=2,
        xlabel=r"$\alpha_{\mathrm{NP}}$", ylabel=r"$\Delta \chi^2$",
        plotname="mc_result", plotitle=None,
        xlims=[None, None], ylims=[None, None], show=False):
    """
    Draw 2-dimensional scatter plot showing the likelihood associated with the
    parameter values given in alphalist. If the lists were computed for multiple
    X-coefficients, the argument x can be used to access a given set of samples.
    The resulting plot is saved in plots directory under plotname.

    """
    fig, ax = plt.subplots()

    nblocks = len(alphas)

    alphalinspace = np.linspace(np.min(np.array(alphas)),
        np.max(np.array(alphas)), 100000)

    if lb is not None and ub is not None:
        ax.axvspan(lb, ub, alpha=.5, color="darkgreen",
        label=f"{nsigmas}" + r"$\sigma$ confidence interval")

    if siglb is not None and sigub is not None:
        ax.axvspan(lb - siglb, lb + siglb, alpha=.2, color="darkgreen")
        ax.axvspan(ub - sigub, ub + sigub, alpha=.2, color="darkgreen")

    ax.axhline(y=delchisqcrit, color="orange", lw=1, ls="--")

    for block in range(nblocks):
        ax.scatter(alphas[block], delchisqs[block],
            s=1, alpha=0.5, color="royalblue")

        ll_fit = parabola(alphalinspace, *parabolaparams[block])

        ax.plot(alphalinspace, ll_fit, color='red', lw=1, ls="--")

        ax.scatter(alphas[block][np.argmin(delchisqs[block])],
            np.min(delchisqs[block]), color='royalblue')

    ax.scatter(bestalphapt, 0, color='red', marker="o",
            label=("best $\\alpha_{\\mathrm{NP}}$ point: "
                + f"{bestalphapt:.4e}"))
    ax.errorbar(bestalphapt, 0, xerr=sigbestalphapt, color="red")

    ax.scatter(bestalphaparabola, 0, color='orange', marker="*",
            label=("best $\\alpha_{\\mathrm{NP}}$ parabola: "
                + f"{bestalphaparabola:.4e}"))
    ax.errorbar(bestalphaparabola, 0, xerr=sigbestalphaparabola, color='orange')

    ax.set_title(plotitle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    plt.legend()
    plt.savefig(_plot_path + "/" + plotname + ".pdf")

    return ax


def plot_parabolic_fit(alphas, ll, params, plotname):
    """Plot generated data and parabolic fit ."""

    plt.figure(figsize=(10, 10 * 6 / 8))
    plt.scatter(alphas, ll, color="orange")
    plt.plot(alphas, parabola(alphas, *params), color="k", lw=1.5)
    plt.savefig(_plot_path + "/parabola_" + plotname + ".pdf")
