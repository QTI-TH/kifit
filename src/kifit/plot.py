import logging

import numpy as np
import matplotlib.pyplot as plt

from kifit.fitools import (
    get_delchisq_crit,
    get_odr_residuals,
    linfit,
    perform_linreg,
    perform_odr,
    collect_fit_X_data
)

from kifit.detools import get_minpos_maxneg_alphaNP_bounds, collect_det_X_data

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

fit_colour = 'orange'
gkp_colour = 'blue'
nmgkp_colour = 'darkgreen'
det_colour = "royalblue"


markerlist = ['o', 'v', '^', '<', '>', 's', 'D']  # maybe cycle?
###############################################################################


def plot_linfit(elem, messenger, magnifac=1, resmagnifac=1, plot_path=None):
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
    plt.savefig(messenger.paths.generate_plot_path("linfit", elemid=elem.id))

    return fig, ax1, ax2, ax3


def blocking_plot(
        messenger,
        nblocks,
        estimations,
        uncertainties,
        label="",
        plotname="blocking_plot"):
    """Plot the blocking iterative estimation of the given list of variables."""

    plt.figure(figsize=(6, 6 * 6 / 8))
    plt.errorbar(np.arange(1, nblocks + 1, 1), estimations, yerr=uncertainties,
        label=label, color="blue", alpha=0.6)
    plt.legend()
    plt.grid(True)
    plt.xlabel("Blocks")
    plt.ylabel("Estimation")

    plotpath = messenger.paths.generate_plot_path(plotname)
    plt.savefig(plotpath)
    logging.info(f"Saving blocking plot to {plotpath}")


def plot_mc_output(
        messenger,
        alphalist,
        delchisqlist,
        minll=0,
        plotname=None,
        xind=0
):
    """
    plot 2-dimensional scatter plot showing the likelihood associated with the
    parameter values given in alphalist.
    The resulting plot is saved in plots directory under plotname.

    """
    fig, ax = plt.subplots()

    nsamples = len(alphalist)  # NEW

    ax.scatter(alphalist, delchisqlist, s=1, alpha=0.5, color=fit_colour)

    smalll_indices = np.argsort(delchisqlist)  # NEW
    small_alphas = np.array([alphalist[ll] for ll in smalll_indices[: int(nsamples * .1)]])  # NEW

    new_alpha = np.median(small_alphas)  # NEW

    ax.axvline(x=new_alpha, color="red", lw=1, ls="-",
            label=rf"new $\alpha$: {new_alpha:.1e}, minll: {minll:.1e}")  # NEW

    ax.set_xlabel(r"$\alpha_{\mathrm{NP}} / \alpha_{\mathrm{EM}}$")
    ax.set_ylabel(r"$\Delta \chi^2$")
    plt.legend(loc='upper center')
    plt.title(rf"{len(alphalist)} samples")

    plotpath = messenger.paths.generate_plot_path("mc_output_" + plotname, xind=xind)
    plt.savefig(plotpath)
    logging.info(f"Saving mc output plot to {plotpath}")

    plt.close()

    return ax


def plot_alphaNP_det_bounds(
    ax,
    messenger,
    gkp,
    dimindex,
    dim,
    xind,
    scatterpos
):
    """
    Plot GKP/NMGKP bounds for one dimension dim.

    """
    det_output = messenger.paths.read_det_output(gkp=gkp, dim=dim, x=xind)

    if det_output['gkp']:
        method_tag = "GKP"
        plot_colour = gkp_colour

    else:
        method_tag = "NMGKP"
        plot_colour = nmgkp_colour

    alphas = det_output['alphas']
    sigalphas = det_output['sigalphas']
    nsigmas = det_output['nsigmas']
    assert dim == det_output['dim']

    (
        minpos, maxneg, allpos, allneg
    ) = get_minpos_maxneg_alphaNP_bounds(
        alphas, sigalphas, nsigmas
    )

    minpos_num = np.nan_to_num(minpos, nan=10.)
    maxneg_num = np.nan_to_num(maxneg, nan=-10.)

    if messenger.params.showalldetbounds:

        for p in range(alphas.shape[0]):  # 1]):
            ax.scatter(
                (allpos.T)[p], scatterpos * np.ones(len((allpos.T)[p])),
                s=0.5,
                color=plot_colour
            )

            ax.scatter(
                (allneg.T)[p], scatterpos * np.ones(len((allneg.T)[p])),
                s=0.5,
                color=plot_colour,
            )

    if messenger.params.showbestdetbounds:

        ax.axvspan(maxneg_num, minpos_num, alpha=.5, color=plot_colour,
            label=("dim-" + str(dim) + " "
                + method_tag
                + f" {nsigmas}" + r"$\sigma$ bounds: "
                + r"$\alpha_{\mathrm{NP}}\in$ ["
                + (f"{maxneg:.1e}" if not np.isnan(maxneg) else "-")
                + ", "
                + (f"{minpos:.1e}" if not np.isnan(minpos) else "-")
                + "]"))

    return ax, minpos, maxneg


def plot_alphaNP_ll(
    elem_collection,
    messenger,
    xlabel=r"$\alpha_{\mathrm{NP}}$",
    ylabel=r"$\Delta \chi^2$",
    xlims=[None, None],
    ylims=[None, None],
    xind=0
):
    """
    Plot 2-dimensional scatter plot showing the likelihood associated with the
    parameter values given in alphalist. If the lists were computed for multiple
    X-coefficients, the argument x can be used to access a given set of samples.
    The resulting plot is saved in plots directory under alphaNP_ll + elem.

    """
    gkpdims = messenger.params.gkp_dims
    nmgkpdims = messenger.params.nmgkp_dims

    elem_collection.check_det_dims(gkpdims, nmgkpdims)

    mc_output = messenger.paths.read_fit_output(xind)

    nsigmas = mc_output['nsigmas']
    delchisqcrit = get_delchisq_crit(nsigmas)

    alphas = mc_output['alphas_exp']
    delchisqs = mc_output['delchisqs_exp']

    nblocks = alphas.shape[0]
    nexps = alphas.shape[1]
    nsamples = alphas.shape[2]

    lb = mc_output['LB']
    ub = mc_output['UB']
    siglb = mc_output['sig_LB']
    sigub = mc_output['sig_UB']

    best_alpha = mc_output['best_alpha']
    sig_best_alpha = mc_output['sig_best_alpha']

    fig, ax = plt.subplots()

    if lb is not None and ub is not None:
        ax.axvspan(lb, ub, alpha=.2, color=fit_colour,
                label=(
                    f"fit {nsigmas}"
                    + r"$\sigma$ confidence interval: $\alpha_{\mathrm{NP}}\in$"
                    + f"[{lb:.1e},{ub:.1e}]"))

    if siglb is not None and sigub is not None:
        ax.axvspan(lb - siglb, lb + siglb, alpha=.2, color=fit_colour)
        ax.axvspan(ub - sigub, ub + sigub, alpha=.2, color=fit_colour)

    ax.axhline(y=delchisqcrit, color="orange", lw=1, ls="--")

    for block in range(nblocks):
        for exp in range(nexps):
            ax.scatter(alphas[block][exp], delchisqs[block][exp],
                s=1, alpha=.2, color=fit_colour)
            ax.scatter(alphas[block][exp][np.argmin(delchisqs[block][exp])],
                np.min(delchisqs[block][exp]), color=fit_colour)

    (ymin, ymax) = ax.get_ylim()

    plotitle = elem_collection.id + ", " + str(nsamples) + " samples, x=" + str(xind)
    ax.set_title(plotitle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlims[0], xlims[1])
    ax.set_ylim(ylims[0], ylims[1])

    errorbarpos = - (ymax - ymin) / 10
    scatterpos = errorbarpos / 2

    ax.errorbar(best_alpha, errorbarpos, xerr=sig_best_alpha, color="red")
    ax.scatter(best_alpha, errorbarpos,
        color="orange", marker="*",
        label=("best fit point: $\\alpha_{\\mathrm{NP}}$="
            + f"{best_alpha:.1e}({sig_best_alpha:.1e})"))

    ax.set_ylim(2 * errorbarpos, ymax)

    for d, dim in enumerate(gkpdims):

        ax, minpos, maxneg = plot_alphaNP_det_bounds(
            ax,
            messenger,
            gkp=True,
            dimindex=d,
            dim=dim,
            xind=xind,
            scatterpos=scatterpos
        )

    for d, dim in enumerate(nmgkpdims):

        ax, minpos_global, maxneg_global = plot_alphaNP_det_bounds(
            ax,
            messenger,
            gkp=False,
            dimindex=d,
            dim=dim,
            xind=xind,
            scatterpos=scatterpos
        )

    plt.legend(loc='upper center')

    plotpath = messenger.paths.generate_plot_path("alphaNP_ll", xind=xind)
    plt.savefig(plotpath)
    logging.info(f"Saving alphaNP-logL plot to {plotpath}")
    plt.close()

    return fig, ax


def plot_mphi_alphaNP_det_bound(
    ax,
    elem,
    messenger,
    dimindex,
    dim,
    gkp=True,
    ylims=[None, None]
):
    """
    Plot GKP/NMGKP bounds for one dimension dim.

    """

    if gkp:
        method_tag = "GKP"
        det_colour = gkp_colour

    else:
        method_tag = "NMGKP"
        det_colour = nmgkp_colour

    minpos, allpos, maxneg, allneg = collect_det_X_data(messenger, dim=dim, gkp=gkp)

    mphis_det = [elem.mphis[x] for x in messenger.x_vals_det]

    min_ub = min(minpos)
    max_lb = max(maxneg)

    if messenger.params.showalldetbounds is True:

        for p in range(allpos.shape[1]):
            if p == 0:
                scatterlabel = (
                    "all dim-" + str(dim) + " " + method_tag + " solutions")
            else:
                scatterlabel = None

            ax.scatter(
                mphis_det,
                (allpos.T)[p],
                s=3,
                color=det_colour,
                alpha=0.3,
                label=scatterlabel,
            )
            ax.scatter(
                mphis_det,
                (allneg.T)[p],
                s=3,
                color=det_colour,
                alpha=0.3,
            )

    if messenger.params.showbestdetbounds is True:
        ax.fill_between(
            mphis_det,
            minpos,
            ylims[1],
            color=det_colour,
            alpha=.2,
            label="best dim-" + str(dim) + " " + method_tag,
        )
        ax.fill_between(
            mphis_det,
            ylims[0],
            maxneg,
            color=det_colour,
            alpha=.2)

    return ax, min_ub, max_lb


def plot_mphi_alphaNP_fit_bound(
    ax,
    elem_collection,
    messenger,
    ylims
):

    UB, sig_UB, LB, sig_LB, best_alphas, sig_best_alphas = collect_fit_X_data(
        messenger=messenger)

    print("UB", UB)

    min_ub = min(UB)
    max_lb = max(LB)

    mphis_fit = [elem_collection.mphis[x] for x in messenger.x_vals_fit]

    ax.errorbar(mphis_fit, best_alphas, yerr=sig_best_alphas,
        color='orange', ls='none')
    ax.scatter(mphis_fit, best_alphas,
        color='orange', marker="*",
        label=r"best fit $\alpha_{\mathrm{NP}} \pm \sigma[\alpha_{\mathrm{NP}}]$")

    ax.fill_between(
        mphis_fit,
        UB, ylims[1] * np.ones(len(UB)),
        color=fit_colour,
        alpha=.2,
        label=(str(messenger.params.num_sigmas) + r" $\sigma$ fit bounds")
    )
    ax.fill_between(
        mphis_fit,
        UB - sig_UB,
        UB + sig_UB,
        color=fit_colour,
        ls='--',
        label=r"uncertainties on fit bounds"
    )

    ax.fill_between(
        mphis_fit,
        ylims[0] * np.ones(len(LB)),
        LB,
        color=fit_colour,
        alpha=.2
    )
    ax.fill_between(
        mphis_fit,
        LB - sig_LB,
        LB + sig_LB,
        color=fit_colour,
        ls='--'
    )

    return ax, min_ub, max_lb


def set_axes_mphi_alpha_plot(
    ax,
    elem_collection,
    xlims,
    ylims,
    linthreshold,
    minub,
    maxlb,
    xlabel,
    ylabel
):

    if xlims[0] is not None:
        ax.set_xlim(left=xlims[0])

    else:
        ax.set_xlim(left=min(elem_collection.mphis))

    if xlims[1] is not None:
        ax.set_xlim(right=xlims[1])

    else:
        ax.set_xlim(right=max(elem_collection.mphis))

    if ylims[0] is not None:
        ymin = ylims[0]

    else:
        ymin = -1

    ax.set_ylim(bottom=ymin)

    if ylims[1] is not None:
        ymax = ylims[1]

    else:
        ymax = 1

    ax.set_ylim(top=ymax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.axhline(y=0, c="k")

    # x-axis
    ax.set_xscale("log")

    # y-axis
    ax.set_yscale("log")

    if linthreshold is None:
        linlim = 10 ** np.floor(np.log10(
            np.nanmax([np.abs(minub), np.abs(maxlb)])) - 1)

    else:
        linlim = linthreshold
    ax.set_yscale("symlog", linthresh=linlim)
    ax.legend(loc="upper left", fontsize="9")

    ax.set_title(elem_collection.id)

    return ax, ymin, ymax


def plot_mphi_alphaNP(
    elem_collection,
    messenger,
    elem=None,
    xlabel=r"$m_\phi~$[eV]",
    ylabel=r"$\alpha_{\mathrm{NP}} / \alpha_{\mathrm{EM}}$",
    xlims=[None, None],
    ylims=[None, None],
    linthreshold=None
):
    """
    Plot the most stringent nsigmas-bounds on both positive and negative
    alphaNP, derived using the Generalised King-plot formula of dimensions d
    listed in dims and save the output under plotname in the plots directory.
    If showall=True, all bounds GKP bounds of the appropriate dimensions are
    shown.

    """
    fig, ax = plt.subplots()

    if ylims[0] is not None:
        ymin = ylims[0]
    else:
        ymin = -1

    if ylims[1] is not None:
        ymax = ylims[1]
    else:
        ymax = 1

    # fit

    ###########################################################################
    ###########################################################################
    minub = ymax
    maxlb = ymin

    if len(messenger.x_vals_fit) > 2:
        ax, minub, maxlb = plot_mphi_alphaNP_fit_bound(
            ax,
            elem_collection,
            messenger,
            ylims=[ymin, ymax])

    # determinant methods
    ###########################################################################

    gkpdims = messenger.params.gkp_dims
    nmgkpdims = messenger.params.nmgkp_dims

    if elem is not None:
        for d, dim in enumerate(gkpdims):
            ax, minub_det, maxlb_det = plot_mphi_alphaNP_det_bound(
                ax,
                elem,
                messenger,
                d,
                dim,
                gkp=True,
                ylims=[ymin, ymax]
            )
            if minub_det < minub:
                minub = minub_det
            if maxlb_det > maxlb:
                maxlb = maxlb_det

        for d, dim in enumerate(nmgkpdims):
            ax, minpos, maxneg = plot_mphi_alphaNP_det_bound(
                ax,
                elem,
                messenger,
                d + len(gkpdims),
                dim,
                gkp=False,
                ylims=[ymin, ymax]
            )
            if minub_det < minub:
                minub = minub_det
            if maxlb_det > maxlb:
                maxlb = maxlb_det

    # formatting + plotting combined det bound
    ###########################################################################
    # gkp_label = (
    #     ("(" + ", ".join(str(gd) for gd in gkpdims) + ")-dim GKP")
    #     if len(gkpdims) > 0
    #     else ""
    # )
    # nmgkp_label = (
    #     ("(" + ", ".join(str(nmd) for nmd in nmgkpdims) + ")-dim NMGKP")
    #     if len(nmgkpdims) > 0
    #     else ""
    # )
    # label_coupling = " + " if (gkp_label != "" and nmgkp_label != "") else ""
    #
    # det_label = gkp_label + label_coupling + nmgkp_label

    print("minub", minub)
    print("maxlb", maxlb)

    ax, ymin, ymax = set_axes_mphi_alpha_plot(
        ax,
        elem_collection,
        xlims,
        ylims,
        linthreshold,
        minub,
        maxlb,
        xlabel,
        ylabel
    )
    #
    # ax.plot(elem.mphis, minpos, color=det_colour)  # , label=det_label)
    # ax.plot(elem.mphis, maxneg, color=det_colour)
    # ax.scatter(elem.mphis, minpos, color=det_colour, s=1)
    # ax.scatter(elem.mphis, maxneg, color=det_colour, s=1)
    #
    # ax.fill_between(
    #     elem.mphis,
    #     minpos,
    #     ymax,
    #     color=det_colour,
    #     alpha=0.3,
    #     label=det_label + " "
    #     + str(messenger.params.num_sigmas) + r"$\sigma$-excluded",
    # )
    # ax.fill_between(
    #     elem.mphis,
    #     maxneg,
    #     ymin,
    #     color=det_colour,
    #     alpha=0.3
    # )
    #
    ax.legend(loc="upper left", fontsize="9")

    plotpath = messenger.paths.generate_plot_path("mphi_alphaNP")
    fig.savefig(plotpath)
    logging.info(f"Saving mphi-alphaNP plot to {plotpath}")

    plt.close()

    return fig, ax
