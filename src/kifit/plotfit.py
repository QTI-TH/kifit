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
fit_colour2 = 'limegreen'

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
    plt.savefig(_plot_path + "/linfit_" + elem.id + ".png")

    return fig, ax1, ax2, ax3


def blocking_plot(nblocks, estimations, uncertainties, label="", filename="blocking"):
    """Plot the blocking iterative estimation of the given list of variables."""
    plt.figure(figsize=(6, 6 * 6 / 8))
    plt.errorbar(np.arange(1, nblocks + 1, 1), estimations, yerr=uncertainties,
        label=label, color="blue", alpha=0.6)
    plt.legend()
    plt.grid(True)
    plt.xlabel("Blocks")
    plt.ylabel("Estimation")
    plt.savefig(f"plots/{filename}.png")


def plot_mc_output(alphalist, delchisqlist,
        nsigmas=2,
        xlabel=r"$\alpha_{\mathrm{NP}} / \alpha_{\mathrm{EM}}$",
        ylabel=r"$\Delta \chi^2$",
        plotname=None,
        xlims=[None, None], ylims=[None, None], llmin=0):
    """
    plot 2-dimensional scatter plot showing the likelihood associated with the
    parameter values given in alphalist.
    The resulting plot is saved in plots directory under plotname.

    """
    fig, ax = plt.subplots()

    nsamples = len(alphalist)  # NEW

    ax.scatter(alphalist, delchisqlist, s=1, alpha=0.5, color="royalblue")

    smalll_indices = np.argsort(delchisqlist)  # NEW
    small_alphas = np.array([alphalist[ll] for ll in smalll_indices[: int(nsamples * .1)]])  # NEW

    new_alpha = np.median(small_alphas)  # NEW

    ax.axvline(x=new_alpha, color="red", lw=1, ls="-",
            label=rf"new $\alpha$: {new_alpha:.4e}")  # NEW

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    plt.legend(loc='upper center')
    plt.title(rf"{len(alphalist)} samples")
    plt.savefig(_plot_path + "/mc_output_" + plotname + ".png")
    plt.close()

    return ax


def plot_final_mc_output(elem, alphas, delchisqs, delchisqcrit,
        bestalphapt=None, sigbestalphapt=None,
        lb=None, siglb=None, ub=None, sigub=None,
        nsigmas=2, xind=0,
        xlabel=r"$\alpha_{\mathrm{NP}}$", ylabel=r"$\Delta \chi^2$",
        plotname="mc_result", plotitle=None,
        xlims=[None, None], ylims=[None, None], show=False):
    """
    Plot 2-dimensional scatter plot showing the likelihood associated with the
    parameter values given in alphalist. If the lists were computed for multiple
    X-coefficients, the argument x can be used to access a given set of samples.
    The resulting plot is saved in plots directory under plotname.

    """
    fig, ax = plt.subplots()

    nexps = len(alphas)
    nsamples = len(alphas[0])

    # alphalinspace = np.linspace(np.min(np.array(alphas)),
    #     np.max(np.array(alphas)), 100000)
    #
    if lb is not None and ub is not None:
        ax.axvspan(lb, ub, alpha=.5, color="darkgreen",
        label=f"{nsigmas}" + r"$\sigma$ confidence interval")

    if siglb is not None and sigub is not None:
        ax.axvspan(lb - siglb, lb + siglb, alpha=.2, color="darkgreen")
        ax.axvspan(ub - sigub, ub + sigub, alpha=.2, color="darkgreen")

    ax.axhline(y=delchisqcrit, color="orange", lw=1, ls="--")

    for exp in range(nexps):
        ax.scatter(alphas[exp], delchisqs[exp],
            s=1, alpha=0.5, color='royalblue')

        ax.scatter(alphas[exp][np.argmin(delchisqs[exp])],
            np.min(delchisqs[exp]), color='royalblue')

    ax.scatter(bestalphapt, 0, color='orange', marker="*",
            label=("best $\\alpha_{\\mathrm{NP}}$ point: "
                + f"{bestalphapt:.4e}"))
    ax.errorbar(bestalphapt, 0, xerr=sigbestalphapt, color="red")

    if plotitle is None:
        plotitle = elem.id + ", " + str(nsamples) + " samples, x=" + str(xind)

    ax.set_title(plotitle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    plt.legend()
    plt.savefig(_plot_path + "/" + plotname + "_" + elem.id + "_x" + str(xind)
        + ".png")
    plt.close()

    return fig, ax


def plot_alphaNP_ll(elem, mc_output, nsigmas: int = 2, xind: int = 0,
    plotname="alphaNP_ll", plotitle=None,
    xlabel=r"$\alpha_{\mathrm{NP}}$", ylabel=r"$\Delta \chi^2$",
        xlims=[None, None], ylims=[None, None]):
    """
    Plot 2-dimensional scatter plot showing the likelihood associated with the
    parameter values given in alphalist. If the lists were computed for multiple
    X-coefficients, the argument x can be used to access a given set of samples.
    The resulting plot is saved in plots directory under alphaNP_ll + elem.

    """

    nsigmas = mc_output[1]

    alphas = mc_output[0][xind][0]
    delchisqs = mc_output[0][xind][1]
    delchisqcrit = mc_output[0][xind][2]
    bestalphapt = mc_output[0][xind][3]
    sigbestalphapt = mc_output[0][xind][4]
    lb = mc_output[0][xind][5]
    siglb = mc_output[0][xind][6]
    ub = mc_output[0][xind][7]
    sigub = mc_output[0][xind][8]

    delchisqcrit = get_delchisq_crit(nsigmas=nsigmas)

    fig, ax = plt.subplots()
    ax.scatter(alphas, delchisqs, s=1, c="b")

    nblocks = len(alphas)
    nsamples = len(alphas[0])

    # alphalinspace = np.linspace(np.min(np.array(alphas)),
    #     np.max(np.array(alphas)), 100000)

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

        ax.scatter(alphas[block][np.argmin(delchisqs[block])],
            np.min(delchisqs[block]), color='royalblue')

    ax.errorbar(bestalphapt, 0, xerr=sigbestalphapt, color="red")
    ax.scatter(bestalphapt, 0,
        color="orange", marker="*",
        label=("best $\\alpha_{\\mathrm{NP}}$ point: "
            + f"{bestalphapt:.4e}({sigbestalphapt:.4e})"))

    if plotitle is None:
        plotitle = elem.id + ", " + str(nsamples) + " samples, x=" + str(xind)
    ax.set_title(plotitle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])
    plt.legend()
    plt.savefig(_plot_path + "/" + plotname + "_" + elem.id + "_x" + str(xind)
        + ".png")
    plt.close()

    return fig, ax


    # fig, ax = plt.subplots()
    # ax.scatter(alphalist[x], delchisqlist_x, s=1, c="b")
    #
    # if len(parabolaparams) == 3:
    #     delchisqcrit_x = get_delchisq_crit(nsigmas=nsigmas)
    #     parampos_x = get_confint(min(alphalist[x]), max(alphalist[x]),
    #         parabolaparams, nsigmas)
    #     ax.axvspan(
    #         np.min(parampos_x), np.max(parampos_x), alpha=0.5, color=fit_colour
    #     )
    #     print("delchisqcrit", delchisqcrit_x)
    #     print("parampos", np.min(parampos_x))
    #     ax.axhline(y=delchisqcrit_x, color="orange", lw=1, ls="--")
    #
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    # ax.set_xlim(xlims[0], xlims[1])
    # ax.set_ylim(ylims[0], ylims[1])
    #
    # if plotname == "":
    #     prettyplotname = ""
    # else:
    #     prettyplotname = plotname + "_"
    #
    # plt.savefig(
    #     _plot_path
    #     + "/alphaNP_ll_"
    #     + elem.id
    #     + "_"
    #     + prettyplotname
    #     + "x"
    #     + str(x)
    #     + "_"
    #     + str(len(alphalist[0]))
    #     + "_fit_samples.png"
    # )
    # return fig, ax


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

    print("minpos", minpos)
    print("maxneg", maxneg)
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


def plot_mphi_alphaNP_fit_bound(ax1, ax2, elem,
    bestalphas_parabola, sigbestalphas_parabola,
    bestalphas_pts, sigbestalphas_pts,
    lb, siglb, ub, sigub,
        plotabs=True, showallowedfitpts=False):

    print("sigbestalphas_parabola", sigbestalphas_parabola)
    print("sigbestalphas_pts", sigbestalphas_pts)

    # mphi vs abs(alphaNP)
    ax1.scatter(elem.mphis, np.abs(bestalphas_parabola), color='orange', marker="*",
        label=r"best $\alpha_{\mathrm{NP}} \pm \sigma[\alpha_{\mathrm{NP}}]$")

    ax1.errorbar(elem.mphis, bestalphas_parabola, yerr=sigbestalphas_parabola,
        color='orange', ls='none')
    # ax1.scatter(elem.mphis, np.abs(bestalphas_pts), color='k', marker="o", s=2)
    # ax1.errorbar(elem.mphis, bestalphas_pts, yerr=sigbestalphas_pts, ecolor='k')

    ax1.plot(elem.mphis, np.max(np.array([np.abs(lb), np.abs(ub)]), axis=0),
        color=fit_colour)
    ax1.plot(elem.mphis,
        np.max(np.array([np.abs(lb) + 2 * siglb, np.abs(ub) + 2 * sigub]), axis=0),
        ls='--', color=fit_colour2, label=r"$2\sigma$ uncertainty on fit bound")

    # mphi vs alphaNP
    ax2.scatter(elem.mphis, bestalphas_parabola, color='orange', marker="*",
        label=r"best $\alpha_{\mathrm{NP}} \pm \sigma[\alpha_{\mathrm{NP}}]$")
    ax2.errorbar(elem.mphis, bestalphas_parabola, yerr=sigbestalphas_parabola,
        color='orange', ls='none')
    # ax2.scatter(elem.mphis, bestalphas_pts, color='k', marker="o", s=2)
    # ax2.errorbar(elem.mphis, bestalphas_pts, yerr=sigbestalphas_pts, ecolor='k')

    ax2.plot(elem.mphis, ub, color=fit_colour)
    ax2.plot(elem.mphis, ub + sigub, ls='--', color=fit_colour2,
        label=r"$2\sigma$ uncertainty on fit bound")

    ax2.plot(elem.mphis, lb, color=fit_colour)
    ax2.plot(elem.mphis, lb - siglb, ls='--', color=fit_colour2)

    print("ub", ub)
    print("lb", lb)

    return ax1, ax2


def set_axes(
    ax1,
    ax2,
    xlims,
    ylims,
    linthreshold,
    elem,
    minpos,
    maxneg,
    absb,
    ub,
    lb,
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
        ymin_ax1 = 1e-17
        ymin_ax2 = -1
    #     ymin_ax1 = (
    #         np.nanmin(np.abs(np.array(list(maxneg) + list(minpos) + list(absb)))) / 10)
    #     ymin_ax2 = np.nanmin(list(maxneg) + list(lb))
    #
    ax1.set_ylim(bottom=ymin_ax1)
    ax2.set_ylim(bottom=ymin_ax2)

    if ylims[1] is not None:
        ymax = ylims[1]

    else:
        ymax = 1
    #     ymax = 10 * np.nanmax(list(minpos) + list(absb))
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
                            # np.nanmin(np.abs(minpos)),
                            # np.nanmin(np.abs(maxneg)),
                            np.nanmin(np.abs(ub)),
                            np.nanmin(np.abs(lb)),
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
    mc_output,
    gkpdims,
    nmgkpdims,
    ndetsamples,
    nsigmas=2,
    plotabs=True,
    plotname="", plotitle="",
    ylabel=r"$\alpha_{\mathrm{NP}} / \alpha_{\mathrm{EM}}$",
    xlims=[None, None], ylims=[None, None],
    linthreshold=None,
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
    # x-vectors
    nsigmas = mc_output[1]
    # alphas = mc_output[0].T[0]
    alphas = np.array([row[0] for row in mc_output[0]])
    # bestalphas_parabola = mc_output[0].T[4]
    bestalphas_parabola = np.array([row[5] for row in mc_output[0]])
    # sigbestalphas_parabola = mc_output[0].T[5]
    sigbestalphas_parabola = np.array([row[6] for row in mc_output[0]])
    # bestalphas_pts = mc_output[0].T[6]
    bestalphas_pts = np.array([row[7] for row in mc_output[0]])
    # sigbestalphas_pts = mc_output[0].T[7]
    sigbestalphas_pts = np.array([row[8] for row in mc_output[0]])
    # lb = mc_output[0].T[8]
    lb = np.array([row[9] for row in mc_output[0]])
    # siglb = mc_output[0].T[9]
    siglb = np.array([row[10] for row in mc_output[0]])
    # ub = mc_output[0].T[10]
    ub = np.array([row[11] for row in mc_output[0]])
    # sigub = mc_output[0].T[11]
    sigub = np.array([row[12] for row in mc_output[0]])

    absb = np.max(np.array([np.abs(lb), np.abs(ub)]), axis=0)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    # fit
    ###########################################################################
    ax1, ax2 = plot_mphi_alphaNP_fit_bound(
        ax1, ax2, elem,
        bestalphas_parabola, sigbestalphas_parabola,
        bestalphas_pts, sigbestalphas_pts,
        lb, siglb, ub, sigub,
        plotabs=plotabs,
        showallowedfitpts=showallowedfitpts)

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
        absb,
        ub,
        lb,
    )

    if len(minpos) > 1 and len(maxneg) > 1:
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
        ax2.fill_between(
            elem.mphis,
            maxneg,
            ymin_ax2,
            color=det_colour,
            alpha=0.3
        )

    if len(absb) > 1:
        ax1.fill_between(
            elem.mphis,
            absb,
            ymax,
            color=fit_colour,
            alpha=0.3,
            label="fit " + str(nsigmas) + r"$\sigma$-excluded",
        )
        ax2.fill_between(
            elem.mphis,
            ub,
            ymax,
            color=fit_colour,
            alpha=0.3,
            label="fit " + str(nsigmas) + r"$\sigma$-excluded",
        )
        ax2.fill_between(
            elem.mphis,
            lb,
            ymin_ax2,
            color=fit_colour,
            alpha=0.3
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
        + str(len(alphas[0]))
        + "_fit_samples.png"
    )

    fig2.savefig(
        _plot_path
        + "/mphi_alphaNP_"
        + elem.id
        + "_"
        + prettyplotname
        + str(len(alphas[0]))
        + "_fit_samples.png"
    )

    plt.close()

    return fig1, ax1, fig2, ax2
