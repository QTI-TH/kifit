import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, interp1d, splrep

from kifit.performfit import (
    get_all_alphaNP_bounds,
    get_confints,
    get_delchisq,
    get_delchisq_crit,
    get_minpos_maxneg_alphaNP_bounds,
    get_odr_residuals,
    interpolate_mphi_alphaNP_fit,
    linfit,
    linfit_x,
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


# lighten & darken colours
##############################################################################


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


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


def plot_alphaNP_ll(
    elem,
    alphalist,
    llist,
    x=0,
    confints=True,
    nsigmas=[1, 2],
    dof=1,
    gkpdims=[],
    nmgkpdims=[],
    plotname="",
    xlabel=r"$\alpha_{\mathrm{NP}}$",
    ylabel=r"$\Delta \chi^2$",
    xlims=[None, None],
    ylims=[None, None],
    show=False,
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

    if confints:
        for ns in nsigmas:
            delchisqcrit_x = get_delchisq_crit(nsigmas=ns, dof=dof)
            parampos_x = get_confints(alphalist[x], delchisqlist_x, delchisqcrit_x)
            ax.axvspan(
                np.min(parampos_x), np.max(parampos_x), alpha=0.5, color=fit_colour
            )
            print("delchisqcrit", delchisqcrit_x)
            print("parampos", np.min(parampos_x))
            if ns == 1:
                hlinels = "--"
            else:
                hlinels = "-"
            ax.axhline(y=delchisqcrit_x, color="orange", linewidth=1, linestyle=hlinels)

    # for dim in gkpdims:
    #     mphis, alphas, sigalphas = sample_alphaNP_det(elem, dim, nsamples,
    #         mphivar=True)
    #     alphas = np.array(alphas)  # [x][perm]
    #     sigalphas = np.array(sigalphas)  # [x][perm]
    #
    #     ax.axhline(
    # if len:
    #     for aNP in elem.alphaNP_GKP:
    #         ax.axhline   #continue here

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


def plot_mphi_alphaNP_fit_bound(
    ax1, ax2, elem, alphalist, llist, nsigmas=2, plotabs=True, showallowedfitpts=False
):

    alphamin = []
    alphamax = []

    delchisqcrit = get_delchisq_crit(nsigmas, dof=1)

    for x in range(alphalist.shape[0]):

        delchisqlist_x = get_delchisq(llist[x])
        alphacrit_x = get_confints(alphalist[x], delchisqlist_x, delchisqcrit)

        # get least stringent bounds on alphaNP
        alphamin_x = np.min(alphacrit_x)
        alphamax_x = np.max(alphacrit_x)

        alphamin.append(alphamin_x)
        alphamax.append(alphamax_x)

        if showallowedfitpts:
            allowed_alphas_x = np.array(
                [a for a in alphalist[x] if alphamin_x < a < alphamax_x]
            )  # * elem.dnorm
            ax1.scatter(
                elem.mphis[x] * np.ones(len(allowed_alphas_x)),
                np.abs(allowed_alphas_x),
                color=fit_colour,
                s=1,
            )
            ax2.scatter(
                elem.mphis[x] * np.ones(len(allowed_alphas_x)),
                allowed_alphas_x,
                color=fit_colour,
                s=1,
            )

    # THIS IS PRETTY AD-HOC. MAYBE IMPLEMENT FAMILY OF FITS.
    nsteps = int(np.floor(len(elem.mphis) / 10))
    bininds = np.array_split(np.arange(len(elem.mphis)), nsteps)

    ax1_bin_mphis = []
    ax1_bin_alphas = []

    ax2_binmax_mphis = []
    ax2_binmax_alphas = []

    ax2_binmin_mphis = []
    ax2_binmin_alphas = []

    alphamin = np.array(alphamin)
    alphamax = np.array(alphamax)
    maxabsalphas = np.max([np.abs(alphamin), np.abs(alphamax)], axis=0)

    for inds in bininds:
        ax1_binind = np.argmax(maxabsalphas[inds])
        ax1_bin_mphis.append((elem.mphis[inds])[ax1_binind])
        ax1_bin_alphas.append((maxabsalphas[inds])[ax1_binind])

        ax2_binmax_ind = np.argmax(alphamax[inds])
        ax2_binmax_mphis.append((elem.mphis[inds])[ax2_binmax_ind])
        ax2_binmax_alphas.append((maxabsalphas[inds])[ax2_binmax_ind])

        ax2_binmin_ind = np.argmin(alphamin[inds])
        ax2_binmin_mphis.append((elem.mphis[inds])[ax2_binmin_ind])
        ax2_binmin_alphas.append((maxabsalphas[inds])[ax2_binmin_ind])

    ax1.plot(ax1_bin_mphis, ax1_bin_alphas, color=fit_colour)  # , label=fit_label)
    f1 = interp1d(
        ax1_bin_mphis, ax1_bin_alphas, kind="slinear", fill_value="extrapolate"
    )
    ax1_fitinterpolpts = f1(elem.mphis)
    ax1.plot(elem.mphis, ax1_fitinterpolpts, color="darkgreen", linestyle="--")

    ax2.plot(
        ax2_binmax_mphis, ax2_binmax_alphas, color=fit_colour
    )  # , label=fit_label)
    f2max = interp1d(
        ax2_binmax_mphis, ax2_binmax_alphas, kind="slinear", fill_value="extrapolate"
    )
    ax2max_fitinterpolpts = f2max(elem.mphis)
    ax2.plot(elem.mphis, ax2max_fitinterpolpts, color="darkgreen", linestyle="--")

    ax2.plot(
        ax2_binmin_mphis, ax2_binmin_alphas, color=fit_colour
    )  # , label=fit_label)
    f2min = interp1d(
        ax2_binmin_mphis, ax2_binmin_alphas, kind="slinear", fill_value="extrapolate"
    )
    ax2min_fitinterpolpts = f2min(elem.mphis)
    ax2.plot(elem.mphis, ax2min_fitinterpolpts, color="darkgreen", linestyle="--")

    return (ax1, ax2, ax1_fitinterpolpts, ax2max_fitinterpolpts, ax2min_fitinterpolpts)


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
    show=False
):
    """
    Draw 2-dimensional scatter plot showing the likelihood associated with the
    parameter values given in paramlist. If the lists were computed for multiple
    X-coefficients, the argument x can be used to access a given set of samples.
    The resulting plot is saved in plots directory under plotname.
    """
    delchisqlist = get_delchisq(llist)

    fig, ax = plt.subplots()
    ax.scatter(paramlist, delchisqlist, s=1)

    ax.scatter(paramlist, delchisqlist, s=1, alpha=0.5, color="royalblue")

    if confints:
        for ns in nsigmas:
            delchisqcrit = get_delchisq_crit(nsigmas=ns, dof=dof)
            parampos = get_confints(
                paramlist, delchisqlist, delchisqcrit,
            )
            ax.axvspan(np.min(parampos), np.max(parampos), alpha=0.5, color="darkgreen")
            ax.axvspan(np.min(parampos), np.max(parampos), alpha=0.5, color="red")
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
    #plt.savefig(_plot_path + "/" + plotname + ".pdf")
    plt.legend()
    plt.savefig(_plot_path + "/" + plotname + ".png")
    if show:
        plt.show()
    return 0


def plot_parabolic_fit(alphas, ll, predictions, parabola_a=None):
    """Plot generated data and parabolic fit."""

    plt.figure(figsize=(10, 10*6/8))
    plt.scatter(alphas, ll, color="orange")
    plt.plot(alphas, predictions, color="black", lw=1.5)
    plt.title(parabola_a)
    plt.savefig(f"parabola_{parabola_a}.png")