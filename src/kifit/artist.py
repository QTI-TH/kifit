import os

import numpy as np
import matplotlib.pyplot as plt

from kifit.hunter import (
    get_delchisq_crit,
    get_minpos_maxneg_alphaNP_bounds,
    get_odr_residuals,
    linfit,
    perform_linreg,
    perform_odr,
    sample_alphaNP_det,
)

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

fit_colour = 'C0'
fit_scatter_colour = 'b'
gkp_colour = 'purple'
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
    plotpath = messenger.generate_plot_path("linfit", elemid=elem.id)
    plt.savefig(plotpath)

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

    plt.savefig(messenger.generate_plot_path(plotname))


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

    plotpath = messenger.generate_plot_path("mc_output_" + plotname, xind=xind)
    plt.savefig(plotpath)
    plt.close()

    return ax


def scatter_alphaNP_det_bounds(
    ax,
    dimindex,
    dim,
    det_output,
    messenger,
    scatterpos,
    minpos_global,
    maxneg_global
):
    """
    Plot GKP/NMGKP bounds for one dimension dim.

    """

    if det_output['gkp']:
        method_tag = "GKP"
        scatter_colour = gkp_colour

    else:
        method_tag = "NMGKP"
        scatter_colour = nmgkp_colour

    # alphas = det_results[0]
    # sigalphas = det_results[1]

    alphas = det_output['alphas']
    sigalphas = det_output['sigalphas']
    nsigmas = det_output['nsigmas']
    assert dim == det_output['dim']

    (
        minpos, maxneg, allpos, allneg
    ) = get_minpos_maxneg_alphaNP_bounds(
        alphas, sigalphas, nsigmas
    )

    if messenger.runparams.showalldetbounds:

        for p in range(alphas.shape[0]):  # 1]):
            # if p == 0:
            #     scatterlabel = elem.id + ", dim " + str(dim) + " " + method_tag
            # else:
            #     scatterlabel = None

            ax.scatter(
                (allpos.T)[p], scatterpos * np.ones(len((allpos.T)[p])),
                s=0.5,
                color=scatter_colour
            )

            ax.scatter(
                # alphaNP_LBs[p], scatterpos,
                (allneg.T)[p], scatterpos * np.ones(len((allneg.T)[p])),
                s=0.5,
                color=scatter_colour,
            )

    if messenger.runparams.showbestdetbounds:

        ax.scatter(
            minpos, scatterpos,
            s=6, marker=markerlist[dim - 3],
            color=scatter_colour,
            label=(det_output['elem'] + ", dim " + str(dim) + " "
                + method_tag + " best: "
                + r"$\alpha_{\mathrm{NP}}$: ["
                # + f"[{maxneg:.1e}, {minpos:.1e}]"
                + (f"{maxneg:.1e}, " if not np.isnan(maxneg) else "-")
                + ", "
                + (f"{minpos:.1e}" if not np.isnan(minpos) else "-")
                + "]")
        )

        ax.errorbar(
            minpos, scatterpos,
            # xerr=np.abs(minpos - minpos_alphas),
            color=scatter_colour)
        ax.scatter(
            maxneg, scatterpos,
            s=6,
            color=scatter_colour)
        ax.errorbar(
            maxneg, scatterpos,
            # xerr=np.abs(maxneg_LB_alphas - maxneg_alphas),
            color=scatter_colour)

    if dimindex == 0:
        minpos_global = minpos
        maxneg_global = maxneg

    else:
        minpos_global = np.fmin(minpos_global, minpos)
        maxneg_global = np.fmax(maxneg_global, maxneg)

    return ax, minpos, maxneg


def plot_vlines_det_bounds(
    ax,
    minpos,
    maxneg,
    nsigmas,
    method_tag,
    gkp=True,
    vlinecolour=gkp_colour,
):
    """
    On alphaNP vs. logL plot, plot best GKP / NMGKP bounds.

    """
    minpos_num = np.nan_to_num(minpos, nan=10.)
    maxneg_num = np.nan_to_num(maxneg, nan=-10.)

    ax.axvspan(maxneg_num, minpos_num, alpha=.5, color=vlinecolour,
        label=f"{nsigmas}" + r"$\sigma$ " + method_tag
        + r". best $\alpha_{\mathrm{NP}}$: ["
        + (f"{maxneg:.1e}, " if not np.isnan(maxneg) else "-")
        + ", "
        + (f"{minpos:.1e}" if not np.isnan(minpos) else "-")
        + "]", lw=1)

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
    gkpdims = messenger.runparams.gkp_dims
    nmgkpdims = messenger.runparams.nmgkp_dims

    elem_collection.check_det_dims(gkpdims, nmgkpdims)

    mc_output = messenger.read_fit_output(xind)

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

    ax.scatter(alphas, delchisqs,
        s=1, c=fit_scatter_colour)

    if lb is not None and ub is not None:
        ax.axvspan(lb, ub, alpha=.5, color=fit_colour,
                label=(
                    f"{nsigmas}"
                    + r"$\sigma$ confidence interval: "
                    + f"[{lb:.1e},{ub:.1e}]"))

    if siglb is not None and sigub is not None:
        ax.axvspan(lb - siglb, lb + siglb, alpha=.2, color=fit_colour)
        ax.axvspan(ub - sigub, ub + sigub, alpha=.2, color=fit_colour)

    ax.axhline(y=delchisqcrit, color="orange", lw=1, ls="--")

    for block in range(nblocks):
        for exp in range(nexps):
            ax.scatter(alphas[block][exp], delchisqs[block][exp],
                s=1, alpha=0.5, color=fit_colour)
            ax.scatter(alphas[block][exp][np.argmin(delchisqs[block][exp])],
                np.min(delchisqs[block][exp]), color=fit_scatter_colour)

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
        label=("best $\\alpha_{\\mathrm{NP}}$ point: "
            + f"{best_alpha:.1e}({sig_best_alpha:.1e})"))

    ax.set_ylim(2 * errorbarpos, ymax)
    minpos_global = np.array([])
    maxneg_global = np.array([])

    for d, dim in enumerate(gkpdims):
        det_output = messenger.read_det_output(gkp=True, dim=dim, x=xind)

        ax, minpos_global, maxneg_global = scatter_alphaNP_det_bounds(
            ax,
            d,
            dim,
            det_output,
            messenger,
            scatterpos,
            minpos_global,
            maxneg_global
        )

    if len(gkpdims) > 0:

        if len(gkpdims) > 1:
            meth_tag = " (" + ", ".join(str(gd) for gd in gkpdims) + ")-dim GKP"

        else:
            meth_tag = f"{gkpdims[0]}-dim GKP"

        ax, minpos_global, maxneg_global = plot_vlines_det_bounds(
            ax,
            minpos_global,
            maxneg_global,
            nsigmas=nsigmas,
            method_tag=meth_tag,
            gkp=True,
            vlinecolour=gkp_colour)

    for d, dim in enumerate(nmgkpdims):
        det_output = messenger.read_det_output(gkp=False, dim=dim, x=xind)

        ax, minpos_global, maxneg_global = scatter_alphaNP_det_bounds(
            ax,
            scatterpos,
            det_output,
            messenger,
            scatterpos,
            minpos_global,
            maxneg_global
        )

    if len(nmgkpdims) > 0:

        if len(nmgkpdims) > 1:
            meth_tag = (
                " (" + ", ".join(str(nmd) for nmd in nmgkpdims) + ")-dim NMGKP")
        else:
            meth_tag = f"{nmgkpdims[0]}-dim NMGKP"

        ax, minpos_global, maxneg_global = plot_vlines_det_bounds(
            ax,
            minpos_global,
            maxneg_global,
            nsigmas=nsigmas,
            method_tag=meth_tag,
            gkp=False,
            vlinecolour=nmgkp_colour)

    plt.legend(loc='upper center')

    plotpath = messenger.generate_plot_path("alphaNP_ll", xind=xind)
    plt.savefig(plotpath)
    print("Saving alphaNP-logL plot to ", plotpath)
    plt.close()

    return fig, ax


def plot_mphi_alphaNP_det_bound(
    ax,
    elem,
    dimindex,
    dim,
    nsamples,
    nsigmas,
    minpos_global,
    maxneg_global,
    gkp=True,
    showalldetbounds=False,
    showbestdetbounds=True
):
    """
    Plot GKP/NMGKP bounds for one dimension dim.

    """

    if gkp:
        method_tag = "GKP"

    else:
        method_tag = "NMGKP"
    alphas, sigalphas = sample_alphaNP_det(elem, dim, nsamples, mphivar=True, gkp=gkp)

    (
        minpos, maxneg, allpos, allneg
    ) = get_minpos_maxneg_alphaNP_bounds(
        alphas, sigalphas, nsigmas
    )

    if showalldetbounds:

        for p in range(alphas.shape[1]):
            if p == 0:
                scatterlabel = elem.id + ", dim " + str(dim) + " " + method_tag
            else:
                scatterlabel = None

            ax.scatter(
                elem.mphis,
                (allpos.T)[p],
                s=0.5,
                color=default_colour[dimindex],
                alpha=0.3,
                label=scatterlabel,
            )
            ax.scatter(
                elem.mphis,
                (allneg.T)[p],
                s=0.5,
                color=default_colour[dimindex],
                alpha=0.3,
            )

    if showbestdetbounds:
        ax.scatter(
            elem.mphis,
            minpos,
            s=3,
            color=default_colour[dimindex],
            label=elem.id + ", dim " + str(dim) + " " + method_tag + "_best",
        )
        ax.scatter(
            elem.mphis,
            maxneg,
            s=3,
            color=default_colour[dimindex])

    if dimindex == 0:
        minpos_global = minpos
        maxneg_global = maxneg

    else:
        minpos_global = np.fmin(minpos_global, minpos)
        maxneg_global = np.fmax(maxneg_global, maxneg)

    return ax, minpos, maxneg


def plot_mphi_alphaNP_fit_bound(
    ax,
    elem,
    bestalphas_pts,
    sigbestalphas_pts,
    lb,
    siglb,
    ub,
    sigub
):
    ax.errorbar(elem.mphis, bestalphas_pts, yerr=sigbestalphas_pts,
        color='orange', ls='none')
    ax.scatter(elem.mphis, bestalphas_pts, color='orange', marker="*",
        label=r"best $\alpha_{\mathrm{NP}} \pm \sigma[\alpha_{\mats}}]$")

    ax.plot(elem.mphis, ub, color=fit_colour)
    ax.plot(elem.mphis, ub + sigub, ls='--', color=fit_colour, alpha=.2,
        label=r"$2\sigma$ uncertainty on fit bound")

    ax.plot(elem.mphis, lb, color=fit_colour)
    ax.plot(elem.mphis, lb - siglb, ls='--', color=fit_colour, alpha=.2)

    return ax


def set_axes(
    ax,
    xlims,
    ylims,
    linthreshold,
    elem,
    minpos,
    maxneg,
    absb,
    ub,
    lb
):

    if xlims[0] is not None:
        ax.set_xlim(left=xlims[0])

    else:
        ax.set_xlim(left=min(elem.mphis))

    if xlims[1] is not None:
        ax.set_xlim(right=xlims[1])

    else:
        ax.set_xlim(right=max(elem.mphis))

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

    ax.set_xlabel(r"$m_\phi~$[eV]")
    ax.set_ylabel(r"$\alpha_{\mathrm{NP}}/\alpha_{\mathrm{EM}}$")

    ax.axhline(y=0, c="k")

    # x-axis
    ax.set_xscale("log")

    # y-axis
    ax.set_yscale("log")

    if linthreshold is None:
        linlim = 10 ** (
            np.floor(
                np.log10(
                    np.nanmin(
                        [
                            np.nanmin(np.abs(ub)),
                            np.nanmin(np.abs(lb)),
                        ]
                    )
                )
                - 1
            )
        )

    else:
        linlim = linthreshold
    ax.set_yscale("symlog", linthresh=linlim)

    return ax, ymin, ymax


def plot_mphi_alphaNP(
    elem_collection,
    elem,
    messenger,
    lb_fit,
    sig_lb_fit,
    ub_fit,
    sig_ub_fit,
    bestalphas,
    sig_bestalphas,
    gkpdims,
    lb_gkp,
    ub_gkp,
    nmgkpdims,
    lb_nmgkp,
    ub_nmgkp,
    nsigmas,
    ylabel=r"$\alpha_{\mathrm{NP}} / \alpha_{\mathrm{EM}}$",
    xlims=[None, None],
    ylims=[None, None],
    linthreshold=None,
    showbestdetbounds=False,
    showalldetbounds=False,
    plot_path=None
):
    """
    Plot the most stringent nsigmas-bounds on both positive and negative
    alphaNP, derived using the Generalised King-plot formula of dimensions d
    listed in dims and save the output under plotname in the plots directory.
    If showall=True, all bounds GKP bounds of the appropriate dimensions are
    shown.

    """

    # nsigmas = mc_output[1]
    #
    # alphas = np.array([row[0] for row in mc_output[0]])
    # delchisqs = np.array([row[1] for row in mc_output[0]])
    # delchisqcrit = mc_output[0, 2]
    # bestalphas_pts = np.array([row[3] for row in mc_output[0]])
    # sigbestalphas_pts = np.array([row[4] for row in mc_output[0]])
    # lb = np.array([row[4] for row in mc_output[0]])
    # siglb = np.array([row[5] for row in mc_output[0]])
    # ub = np.array([row[6] for row in mc_output[0]])
    # sigub = np.array([row[7] for row in mc_output[0]])
    #
    # absb = np.max(np.array([np.abs(lb), np.abs(ub)]), axis=0)

    fig, ax = plt.subplots()


    # fit

    ###########################################################################
    ###########################################################################

    ax = plot_mphi_alphaNP_fit_bound(
        ax,
        elem_collection,
        bestalphas,
        sig_bestalphas,
        lb_fit,
        sig_lb_fit,
        ub_fit,
        sig_ub_fit)

    # determinant methods
    ###########################################################################

    minpos = np.array([])
    maxneg = np.array([])

    for d, dim in enumerate(gkpdims):
        ax, minpos, maxneg = plot_mphi_alphaNP_det_bound(
            ax,
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
        ax, minpos, maxneg = plot_mphi_alphaNP_det_bound(
            ax,
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

    ax, ymin, ymax = set_axes(
        ax,
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

        ax.plot(elem.mphis, minpos, color=det_colour)  # , label=det_label)
        ax.plot(elem.mphis, maxneg, color=det_colour)
        ax.scatter(elem.mphis, minpos, color=det_colour, s=1)
        ax.scatter(elem.mphis, maxneg, color=det_colour, s=1)

        ax.fill_between(
            elem.mphis,
            minpos,
            ymax,
            color=det_colour,
            alpha=0.3,
            label=det_label + " " + str(nsigmas) + r"$\sigma$-excluded",
        )
        ax.fill_between(
            elem.mphis,
            maxneg,
            ymin,
            color=det_colour,
            alpha=0.3
        )

    if len(absb) > 1:

        ax.fill_between(
            elem.mphis,
            ub,
            ymax,
            color=fit_colour,
            alpha=0.3,
            label="fit " + str(nsigmas) + r"$\sigma$-excluded",
        )
        ax.fill_between(
            elem.mphis,
            lb,
            ymin,
            color=fit_colour,
            alpha=0.3
        )

    ax.legend(loc="upper left", fontsize="9")

    if plotname == "":
        prettyplotname = ""
    else:
        prettyplotname = plotname + "_"

    fig.savefig(os.path.join(plot_path,
        f"mphi_alphaNP_{elem.id}_{prettyplotname}{str(len(alphas[0]))}_fit_samples.png"))

    plt.close()

    return fig, ax
