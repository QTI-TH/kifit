import logging

import numpy as np
import matplotlib.pyplot as plt

from kifit.build import get_odr_residuals, linfit, perform_linreg, perform_odr

from kifit.fitools import get_delchisq_crit, collect_fit_X_data

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

mc_scatter_colour = 'C0'
fit_scatter_colour = 'orangered'
fit_colour = 'orange'
gkp_colour = 'blue'
nmgkp_colour = 'darkgreen'
proj_colour = 'purple'
# det_colour = "royalblue"


markerlist = ['o', 'v', '^', '<', '>', 's', 'D']  # maybe cycle?
###############################################################################


def plot_linfit(elem, messenger, magnifac=1, resmagnifac=1, plot_path=None):
    """
    Plot, King plot data and output of linear regression and orthogonal distance
    regression. Use to check linear fit.

    Input: instance of Elem class, plot name, boolean whether or not to show plot
    Output: plot saved in plots directory with name linfit + elem.

    """

    betas_odr, sig_betas_odr, kperp1s, ph1s, sig_kperp1s, sig_ph1s, _ = perform_odr(
        elem.nutil_in,
        elem.sig_nutil_in,
        reference_transition_index=0,
    )

    (
        betas_linreg,
        sig_betas_linreg,
        kperp1s_linreg,
        ph1s_linreg,
        sig_kperp1s_linreg,
        sig_ph1s_linreg,
    ) = perform_linreg(elem.nutil_in,
        reference_transition_index=0)

    xvals = elem.nutil_in.T[0]
    sigxvals = elem.sig_nutil_in.T[0]
    yvals = elem.nutil_in[:, 1:].T
    sigyvals = elem.sig_nutil_in[:, 1:].T

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
    plt.savefig(messenger.paths.generate_plot_path("linfit", elemid=elem.id),
        dpi=1000)

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
    plt.savefig(plotpath, dpi=1000)
    logging.info(f"Saving blocking plot to {plotpath}")


def plot_mc_output(
        messenger,
        alphalist,
        delchisqlist,
        expstr="experiment",
        plotname=None,
        xind=0,
        logplot=False
):
    """
    plot 2-dimensional scatter plot showing the likelihood associated with the
    parameter values given in alphalist.
    The resulting plot is saved in plots directory under plotname.

    """
    fig, ax = plt.subplots()

    ax.scatter(alphalist, delchisqlist, s=1, alpha=0.5, color=mc_scatter_colour)

    ax.set_xlabel(r"$\alpha_{\mathrm{NP}} / \alpha_{\mathrm{EM}}$")
    ax.set_ylabel(r"$\Delta \chi^2$")

    plt.title(f"x={xind}, {len(alphalist)}" + r" $\alpha_{\mathrm{NP}}$ samples")

    plotpath = messenger.paths.generate_plot_path("mc_output_" + plotname, xind=xind)
    plt.savefig(plotpath, dpi=1000)
    logging.info(f"Saving mc output plot to {plotpath}")

    plt.close()

    return ax


def plot_search_output(
        messenger,
        alphalist,
        delchisqlist,
        delchisqcrit,
        searchlims,
        xind=0,
        logplot=False
):
    """
    Plot output of search phase:
    2-dimensional scatter plot showing the likelihood associated with the
    parameter values given in alphalist.
    The resulting plot is saved in plots directory under plotname.

    """
    fig, ax = plt.subplots()

    alphalist = np.array(alphalist).flatten()
    delchisqlist = np.array(delchisqlist).flatten()

    ax.scatter(alphalist, delchisqlist, s=1, alpha=0.5, color=mc_scatter_colour)

    if logplot is True and np.min(alphalist) < 0 and np.max(alphalist) > 0:
        linthresh_x = np.min(np.abs(alphalist))
        ax.set_xscale("symlog", linthresh=linthresh_x)
        ax.set_yscale("log")
        ax.axvline(x=-linthresh_x, color='k', ls='--', label="linear threshold")
        ax.axvline(x=linthresh_x, color='k', ls='--')

    if delchisqcrit is not None:
        ax.axhline(y=delchisqcrit, color="r", ls='--',
            label=r"$\Delta \chi^2\vert_{\mathrm{crit.}}$")

    # alphas_inside = alphalist[np.argwhere(delchisqlist < delchisqcrit).flatten()]
    print("searchlims", (np.min(searchlims), np.max(searchlims)))
    ax.axvline(x=np.min(searchlims), color="orange", ls='--', label="search interval")
    ax.axvline(x=np.max(searchlims), color="orange", ls='--')

    ax.set_xlabel(r"$\alpha_{\mathrm{NP}} / \alpha_{\mathrm{EM}}$")
    ax.set_ylabel(r"$\Delta \chi^2$")

    if not ax.get_legend_handles_labels() == ([], []):
        plt.legend(loc='upper center')

    plt.title(f"x={xind}, {len(alphalist)}" + r" $\alpha_{\mathrm{NP}}$ samples")

    plotpath = messenger.paths.generate_plot_path(
        "search_output_"
        + (f"{messenger.params.search_mode}-search")
        + (f"{messenger.params.logrid_frac}logridfrac_"
            if messenger.params.search_mode == "detlogrid" else ""), xind=xind)
    print("plotpath", plotpath)
    plt.savefig(plotpath, dpi=1000)
    logging.info(f"Saving mc output plot to {plotpath}")

    plt.close()

    return ax


def plot_alphaNP_det_bounds(
    ax1, ax2,
    messenger,
    detstr,
    dimindex,
    dim,
    xind
):
    """
    Plot GKP/NMGKP bounds for one dimension dim.

    """
    det_output = messenger.paths.read_det_output(detstr=detstr, dim=dim, x=xind)

    if det_output['detstr']=='gkp':
        method_tag = "GKP"
        plot_colour = gkp_colour

    elif det_output['detstr']=='nmgkp':
        method_tag = "NMGKP"
        plot_colour = nmgkp_colour

    elif det_output['detstr']=='proj':
        method_tag = "proj"
        plot_colour = proj_colour

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

        ax2.scatter(allpos, np.zeros(len(allpos)),
            s=1, color=plot_colour)

        ax2.scatter(allneg, np.zeros(len(allpos)),
            s=1, color=plot_colour)

        #
        #
        # for p in range(npermutations):  # 1]):
        #     ax.scatter(
        #         (allpos.T)[p], scatterpos * np.ones(len((allpos.T)[p])),
        #         s=0.5,
        #         color=plot_colour
        #     )
        #
        #     ax.scatter(
        #         (allneg.T)[p], scatterpos * np.ones(len((allneg.T)[p])),
        #         s=0.5,
        #         color=plot_colour,
        #     )
    if messenger.params.showbestdetbounds:

        ax1.axvline(x=maxneg_num, ls="--", color=plot_colour,
            label=("dim-" + str(dim) + " "
                + method_tag
                + f" {nsigmas}" + r"$\sigma$ bounds: "
                + r"$\alpha_{\mathrm{NP}}\in$ ["
                + (f"{maxneg:.1e}" if not np.isnan(maxneg) else "-")
                + ", "
                + (f"{minpos:.1e}" if not np.isnan(minpos) else "-")
                + "]"))

        ax1.axvline(x=minpos_num, ls="--", color=plot_colour)

    return ax1, ax2, minpos, maxneg


def plot_alphaNP_ll(
    elem_collection,
    messenger,
    expstr="experiment",
    logplot=False,
    xlabel=r"$\alpha_{\mathrm{NP}}/\alpha_{\mathrm{EM}}$",
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
    projdims = messenger.params.proj_dims

    elem_collection.check_det_dims(gkpdims, nmgkpdims, projdims)

    if expstr == "experiment":
        print("this is an experiment")
        mc_output = messenger.paths.read_fit_output(xind)

        delchisqs = mc_output['delchisqs_exp']
        nsigmas = mc_output['nsigmas']
        delchisqcrit = get_delchisq_crit(nsigmas)
        delchisqcrit_label = r"$\Delta \chi^2_{\mathrm{crit}}$"

    elif expstr == "search":
        print("this is a search")
        mc_output = messenger.paths.read_search_output(xind)

        delchisqs = mc_output['delchisqs_exp']
        delchisqcrit = np.median(delchisqs)
        delchisqcrit_label = r"median $\Delta \chi^2$ samples"

        nsigmas = mc_output['nsigmas']

    alphas = mc_output['alphas_exp']
    nexps = alphas.shape[0]
    nsamples = alphas.shape[1]

    best_alpha = mc_output['best_alpha']
    sig_best_alpha = mc_output['sig_best_alpha']

    lb = mc_output['LB']
    ub = mc_output['UB']
    siglb = mc_output['sig_LB']
    sigub = mc_output['sig_UB']

    fig = plt.figure()
    ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=7)
    ax2 = plt.subplot2grid((8, 1), (7, 0))

    if lb is not None and ub is not None:
        if expstr == "experiment":
            bound_label = (
                f"fit {nsigmas}"
                + r"$\sigma$ confidence interval: $\alpha_{\mathrm{NP}}\in$"
                + f"[{lb:.1e},{ub:.1e}]")

        elif expstr == "search":
            bound_label = "search interval"

        ax1.axvspan(lb, ub, alpha=.2, color=fit_scatter_colour, label=bound_label)

    if siglb is not None and sigub is not None:
        ax1.axvspan(lb - siglb, lb + siglb, alpha=.7, color=fit_scatter_colour)
        ax1.axvspan(ub - sigub, ub + sigub, alpha=.7, color=fit_scatter_colour)

    for exp in range(nexps):
        ax1.scatter(alphas[exp], delchisqs[exp],
            s=1, alpha=.2, color=fit_scatter_colour, zorder=3)
        ax1.scatter(alphas[exp][np.argmin(delchisqs[exp])],
            np.min(delchisqs[exp]), color=fit_scatter_colour, zorder=3)

    ax1.axhline(y=delchisqcrit, color="r", lw=1, ls="--", label=delchisqcrit_label)

    plotitle = elem_collection.id + ", " + str(nsamples) + " samples, x=" + str(xind)
    ax1.set_title(plotitle)
    ax1.set_ylabel(ylabel)
    ax1.set_xlim(xlims[0], xlims[1])
    ax1.set_ylim(ylims[0], ylims[1])
    ax2.set_xlabel(xlabel)
    ax2.set_xlim(xlims[0], xlims[1])
    ax2.set_ylim(ylims[0], ylims[1])

    ax2.errorbar(best_alpha, 0, xerr=sig_best_alpha, color="orange")
    ax2.scatter(best_alpha, 0,
        color="orange", marker="*",
        label=("best fit point: $\\alpha_{\\mathrm{NP}}$="
            + f"{best_alpha:.1e}({sig_best_alpha:.1e})"))

    # if elem_collection.len == 1:
    for d, dim in enumerate(gkpdims):

        ax1, ax2, minpos, maxneg = plot_alphaNP_det_bounds(
            ax1, ax2,
            messenger,
            detstr="gkp",
            dimindex=d,
            dim=dim,
            xind=xind
        )

    for d, dim in enumerate(nmgkpdims):

        ax1, ax2, minpos_global, maxneg_global = plot_alphaNP_det_bounds(
            ax1, ax2,
            messenger,
            detstr="nmgkp",
            dimindex=d,
            dim=dim,
            xind=xind
        )

    for d, dim in enumerate(projdims):

        ax1, ax2, minpos_global, maxneg_global = plot_alphaNP_det_bounds(
            ax1, ax2,
            messenger,
            detstr="proj",
            dimindex=d,
            dim=dim,
            xind=xind
        )

    if not ax1.get_legend_handles_labels() == ([], []):
        ax1.legend(loc='upper center')

    if logplot is True:
        linthresh_x = np.min(np.abs(alphas))
        ax1.set_xscale("symlog", linthresh=linthresh_x)
        ax2.set_xscale("symlog", linthresh=linthresh_x)
        ax1.set_yscale("log")
        ax1.axvline(x=-linthresh_x, color='k', ls='--', label="linear threshold")
        ax1.axvline(x=linthresh_x, color='k', ls='--')
        ax2.axvline(x=-linthresh_x, color='k', ls='--', label="linear threshold")
        ax2.axvline(x=linthresh_x, color='k', ls='--')

        ax1.set_ylim([np.min(delchisqs), np.max(delchisqs)])

    (xmin, xmax) = ax1.get_xlim()
    ax2.set_xlim([xmin, xmax])
    ax2.set_ylim([-.5, .5])
    ax2.set_yticks([])

    ax1.set_title(f"x={xind}, {nsamples}" + r" $\alpha_{\mathrm{NP}}$ samples")

    plotpath = messenger.paths.generate_plot_path(
        "alphaNP_ll"
        + ("_" + expstr if expstr == "search" else "")
        + (f"{messenger.params.search_mode}-search")
        + (f"{messenger.params.logrid_frac}logridfrac_"
            if messenger.params.search_mode == "detlogrid" else ""), xind=xind)

    plt.savefig(plotpath, dpi=1000)

    logging.info(f"Saving alphaNP-logL plot to {plotpath}")
    plt.close()

    return fig, ax1, ax2


def plot_mphi_alphaNP_det_bound(
    ax,
    elem,
    messenger,
    dimindex,
    dim,
    detstr="gkp",
    ylims=[None, None]
):
    """
    Plot GKP/NMGKP bounds for one dimension dim.

    """

    if detstr=="gkp":
        method_tag = "GKP"
        det_colour = gkp_colour

    elif detstr=="nmgkp":
        method_tag = "NMGKP"
        det_colour = nmgkp_colour

    elif detstr=="proj":
        print("hullu proj")
        method_tag = "proj"
        det_colour = proj_colour

    alphas, sigalphas, minpos, allpos, maxneg, allneg = collect_det_X_data(
        messenger, dim=dim, detstr=detstr)
    print("detstr", detstr)
    print("alphas", alphas)

    npermutations = alphas.shape[1]

    mphis_det = [elem.mphis[x] for x in messenger.x_vals_det]
    min_mphis_det = min(mphis_det)
    max_mphis_det = max(mphis_det)

    min_ub = min(minpos)
    max_lb = max(maxneg)

    if messenger.params.showalldetvals is True:

        for p in range(npermutations):
            if p == 0:
                meanvalabel = ("all " + str(dim) + " " + method_tag
                    + r" solutions $\pm 1 \sigma$")
            else:
                meanvalabel = None

            ax.errorbar(
                mphis_det,
                (alphas.T)[p],
                yerr=(sigalphas.T)[p],
                color=det_colour,
                ls=':',
                label=meanvalabel)

    if messenger.params.showalldetbounds is True:

        for p in range(npermutations):
            if p == 0:
                scatterlabel=("all " + str(messenger.params.num_sigmas)
                + r"$\sigma$-bounds dim-" + str(dim)
                + " " + method_tag)

            else:
                scatterlabel = None

            ax.scatter(
                mphis_det,
                (allpos.T)[p],
                s=3,
                color=det_colour,
                label=scatterlabel)

            ax.scatter(
                mphis_det,
                (allneg.T)[p],
                s=3,
                color=det_colour)

    if messenger.params.showbestdetbounds is True:
        ax.plot(  # ax.fill_between(
            mphis_det,
            minpos,  # ylims[1],
            color=det_colour,
            ls='--',  # alpha=.2,
            label=("best " + str(messenger.params.num_sigmas)
                + r"$\sigma$-bounds dim-" + str(dim)
                + " " + method_tag))

        ax.plot(  # ax.fill_between(
            mphis_det,  # ylims[0],
            maxneg,
            ls='--',
            color=det_colour)  # ,alpha=.2)

    return ax, min_ub, max_lb, min_mphis_det, max_mphis_det


def plot_mphi_alphaNP_fit_bound(
    ax,
    elem_collection,
    messenger,
    ylims
):

    UB, sig_UB, LB, sig_LB, best_alphas, sig_best_alphas = collect_fit_X_data(
        messenger=messenger)

    min_ub = np.nanmin(UB)
    max_lb = np.nanmax(LB)

    mphis_fit = [elem_collection.mphis[x] for x in messenger.x_vals_fit]
    min_mphis_fit = min(mphis_fit)
    max_mphis_fit = max(mphis_fit)

    ax.errorbar(mphis_fit, best_alphas, yerr=sig_best_alphas,
        color='orange', ls='none', zorder=1)
    ax.scatter(mphis_fit, best_alphas,
        color='orange', marker="*", zorder=0,
        label=r"best fit $\alpha_{\mathrm{NP}}^* \pm \sigma[\alpha_{\mathrm{NP}}^*]$")

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
        ls='-',
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
        ls='-'
    )

    return ax, min_ub, max_lb, min_mphis_fit, max_mphis_fit


def set_axes_mphi_alpha_plot(
    ax,
    elem_collection,
    config,
    min_mphi,
    max_mphi,
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
        ax.set_xlim(left=min_mphi / 1.4)

    if xlims[1] is not None:
        ax.set_xlim(right=xlims[1])

    else:
        ax.set_xlim(right=max_mphi * 1.4)

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
    ax.axhline(y=linlim, color='k', ls='--')
    ax.axhline(y=-linlim, color='k', ls='--')
    ax.tick_params(axis='both', labelsize=9)
    ax.locator_params(axis='y', numticks=6)
    ax.set_title(elem_collection.id, fontsize=11)

    return ax, ymin, ymax


def plot_mphi_alphaNP(
    elem_collection,
    messenger,
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
    fig, ax = plt.subplots(figsize=(6.8, 3.4))

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
    minub = np.nan
    maxlb = np.nan

    min_mphis_fit = np.nan
    max_mphis_fit = np.nan

    min_mphis_det = np.nan
    max_mphis_det = np.nan

    if len(messenger.x_vals_fit) >= 2:
        (
            ax, minub, maxlb, min_mphis_fit, max_mphis_fit
        ) = plot_mphi_alphaNP_fit_bound(
            ax,
            elem_collection,
            messenger,
            ylims=[ymin, ymax])

    # determinant methods
    ###########################################################################

    gkpdims = messenger.params.gkp_dims
    nmgkpdims = messenger.params.nmgkp_dims
    projdims = messenger.params.proj_dims

    # if elem_collection.len == 1:
    for elem in elem_collection.elems:
        # elem = elem_collection.elems[0]

        for d, dim in enumerate(gkpdims):
            (
                ax, minub_det, maxlb_det, min_mphis_det, max_mphis_det
            ) = plot_mphi_alphaNP_det_bound(
                ax,
                elem,
                messenger,
                d,
                dim,
                detstr="gkp",
                ylims=[ymin, ymax]
            )
            if minub_det < np.nan_to_num(minub, nan=ymax):
                minub = minub_det
            if maxlb_det > np.nan_to_num(maxlb, nan=ymin):
                maxlb = maxlb_det

        for d, dim in enumerate(nmgkpdims):
            (
                ax, minpos, maxneg, min_mphis_det, max_mphis_det
            ) = plot_mphi_alphaNP_det_bound(
                ax,
                elem,
                messenger,
                d + len(gkpdims),
                dim,
                detstr="nmgkp",
                ylims=[ymin, ymax]
            )
            if minub_det < np.nan_to_num(minub, nan=ymax):
                minub = minub_det
            if maxlb_det > np.nan_to_num(maxlb, nan=ymin):
                maxlb = maxlb_det

        for d, dim in enumerate(projdims):
            (
                ax, minpos, maxneg, min_mphis_det, max_mphis_det
            ) = plot_mphi_alphaNP_det_bound(
                ax,
                elem,
                messenger,
                d + len(gkpdims) + len(nmgkpdims),
                dim,
                detstr="proj",
                ylims=[ymin, ymax]
            )
            if minub_det < np.nan_to_num(minub, nan=ymax):
                minub = minub_det
            if maxlb_det > np.nan_to_num(maxlb, nan=ymin):
                maxlb = maxlb_det

    min_mphi = np.nanmax([min_mphis_det, min_mphis_fit])
    max_mphi = np.nanmin([max_mphis_det, max_mphis_fit])

    ax, ymin, ymax = set_axes_mphi_alpha_plot(
        ax,
        elem_collection,
        messenger,
        min_mphi,
        max_mphi,
        xlims,
        ylims,
        linthreshold,
        minub,
        maxlb,
        xlabel,
        ylabel
    )

    ax.legend(fontsize="9", loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plotpath = messenger.paths.generate_plot_path("mphi_alphaNP")
    fig.savefig(plotpath, dpi=1000)
    logging.info(f"Saving mphi-alphaNP plot to {plotpath}")

    plt.close()

    return fig, ax
