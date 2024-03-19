import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# import pandas as pd

from kifit.performfit import (linfit, linfit_x, get_odr_residuals,
    perform_linreg, perform_odr,
    get_delchisq, get_delchisq_crit, get_confints, interpolate_mphi_alphaNP_fit,
    sample_alphaNP_det, get_minpos_maxneg_alphaNP_bounds)

_plot_path = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'plots'
))

if not os.path.exists(_plot_path):
    os.makedirs(_plot_path)

###############################################################################

default_colour = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

det_colour = 'b'
fit_colour = 'darkgreen'


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
    import matplotlib.colors as mc
    import colorsys
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
        elem.mu_norm_isotope_shifts_in, elem.sig_mu_norm_isotope_shifts_in,
        reftrans_index=0)

    (betas_linreg, sig_betas_linreg, kperp1s_linreg, ph1s_linreg,
        sig_kperp1s_linreg, sig_ph1s_linreg) = perform_linreg(
        elem.mu_norm_isotope_shifts_in, reftrans_index=0)

    xvals = elem.mu_norm_isotope_shifts_in.T[0]
    sigxvals = elem.sig_mu_norm_isotope_shifts_in.T[0]
    yvals = elem.mu_norm_isotope_shifts_in[:, 1:].T
    sigyvals = elem.sig_mu_norm_isotope_shifts_in[:, 1:].T

    AAp = np.array([elem.a_nisotope, elem.ap_nisotope]).T

    xmin = 0.95 * min(xvals)
    xmax = 1.05 * max(xvals)
    xfit = np.linspace(xmin, xmax, 1000)

    plt.figure()
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((4, 1), (2, 0))
    ax3 = plt.subplot2grid((4, 1), (3, 0))
    ax1.set_title("King Plot")

    ax2.axhline(y=0, c='k')
    ax3.axhline(y=0, c='k')

    for i in range(yvals.shape[0]):

        yfit_linreg = linfit(betas_linreg[i], xfit)
        residuals_linreg_i, sigresiduals_linreg_i = get_odr_residuals(
            betas_linreg[i], xvals, yvals[i], sigxvals, sigyvals[i])

        yfit_odr_i = linfit(betas_odr[i], xfit)

        residuals_odr_i, sigresiduals_odr_i = get_odr_residuals(betas_odr[i],
            xvals, yvals[i], sigxvals, sigyvals[i])

        ax1.plot(xfit, yfit_odr_i, color=default_colour[i], label="i=" + str(i + 2))

        ax1.plot(xfit, yfit_linreg, color=default_colour[i], linestyle='--')

        for a in range(yvals.shape[1]):
            ax1.annotate(
                "(" + str(int(AAp[a, 0])) + "," + str(int(AAp[a, 1])) + ")",
                (xvals[a], yvals[i, a]), fontsize=8)

        ax1.errorbar(xvals, yvals[i],
            xerr=sigxvals,
            yerr=sigyvals[i], linestyle='none', marker='o', ms=4)

        ax2.errorbar(xvals, magnifac * residuals_odr_i,
            xerr=sigresiduals_odr_i,
            yerr=resmagnifac * magnifac * sigresiduals_odr_i,
            linestyle='none', marker='o', ms=4, capsize=2)

        ax3.errorbar(xvals, magnifac * residuals_odr_i,
            xerr=sigresiduals_odr_i,
            yerr=resmagnifac * magnifac * sigresiduals_odr_i,
            linestyle='none', marker='o', ms=4, capsize=2)

    ax1.set_xlim(xmin, xmax)
    ax1.set_xlabel(r"$\tilde{\nu}_1~$[Hz]")
    ax1.set_ylabel(r"$\tilde{\nu}_i~$[Hz]")
    ax1.legend()
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylabel(r"ODR Resid.")
    ax3.set_xlim(xmin, xmax)
    ax3.set_ylabel(r"Lin. Reg. Resid.")

    plt.tight_layout()
    plt.savefig(_plot_path + "/linfit_" + elem.id + ".pdf")


def plot_alphaNP_ll(elem, alphalist, llist, x=0,
        confints=True, nsigmas=[1, 2], dof=1, gkpdims=[], nmgkpdims=[],
        xlabel=r'$\alpha_{\mathrm{NP}}$', ylabel=r"$\Delta \chi^2$",
        xlims=[None, None], ylims=[None, None], show=False):
    """
    Plot 2-dimensional scatter plot showing the likelihood associated with the
    parameter values given in alphalist. If the lists were computed for multiple
    X-coefficients, the argument x can be used to access a given set of samples.
    The resulting plot is saved in plots directory under alphaNP_ll + elem.

    """

    delchisqlist_x = get_delchisq(llist[x])

    fig, ax = plt.subplots()
    ax.scatter(alphalist[x], delchisqlist_x, s=1, c='b')

    if confints:
        for ns in nsigmas:
            delchisqcrit_x = get_delchisq_crit(nsigmas=ns, dof=dof)
            parampos_x = get_confints(
                alphalist[x], delchisqlist_x, delchisqcrit_x)
            ax.axvspan(np.min(parampos_x), np.max(parampos_x), alpha=.5,
            color=fit_colour)
            print("delchisqcrit", delchisqcrit_x)
            print("parampos", np.min(parampos_x))
            if ns==1:
                hlinels = '--'
            else:
                hlinels = '-'
            ax.axhline(y=delchisqcrit_x, color='orange', linewidth=1,
                linestyle=hlinels)

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
    plt.savefig(_plot_path + "/alphaNP_ll_" + elem.id + "_x" + str(x) + ".pdf")
    if show:
        plt.show()
    return 0


def plot_mphi_alphaNP_det_bound(ax, elem, dimindex, dim, nsamples, nsigmas,
        minpos, maxneg, shrubberymin, shrubberymax,
        gkp=True, showbestdetbounds=False, showalldetbounds=False):
    """
    Plot GKP/NMGKP bounds for one dimension dim.

    """
    if gkp:
        method_tag = "GKP"
    else:
        method_tag = "NMGKP"

    alphas, sigalphas = sample_alphaNP_det(elem, dim, nsamples, mphivar=True,
        gkp=gkp)

    if showalldetbounds:
        for p in range(alphas.shape[1]):
            ax.scatter(elem.mphis, (alphas.T)[p] + nsigmas * (sigalphas.T)[p],
                s=4, color=lighten_color(default_colour[dimindex], 0.5))
    (minpos_alphas, minpos_sigalphas, maxneg_alphas, maxneg_sigalphas) = \
        get_minpos_maxneg_alphaNP_bounds(alphas, sigalphas, nsigmas)

    if showbestdetbounds:
        ax.scatter(elem.mphis, minpos_alphas + nsigmas * minpos_sigalphas,
            s=6, color=default_colour[dimindex],
            label=elem.id + ", dim " + str(dim) + " " + method_tag)

        ax.scatter(elem.mphis, maxneg_alphas - nsigmas * maxneg_sigalphas,
            s=6, color=default_colour[dimindex])

    if dimindex == 0:
        # mphis = mphis
        minpos = minpos_alphas + nsigmas * minpos_sigalphas
        maxneg = maxneg_alphas - nsigmas * maxneg_sigalphas
    else:
        minpos = np.fmin(minpos,
            minpos_alphas + nsigmas * minpos_sigalphas)
        maxneg = np.fmax(maxneg,
            maxneg_alphas - nsigmas * maxneg_sigalphas)

    # update shrubbery
    if 0 < np.nanmin(minpos_alphas) < shrubberymax:
        shrubberymax = np.nanmin(minpos_alphas)
    if 0 > np.nanmax(maxneg_alphas) > shrubberymin:
        shrubberymin = np.nanmax(maxneg_alphas)

    return ax, minpos, maxneg, shrubberymin, shrubberymax


def plot_mphi_alphaNP_fit_bound(ax, elem, alphalist, llist,
        shrubberymin, shrubberymax,
        nsigmas=2, plotabs=True,
        showallowedfitpts=False):

    alphamin = []
    alphamax = []

    delchisqcrit = get_delchisq_crit(nsigmas, dof=1)

    for x in range(alphalist.shape[0]):

        delchisqlist_x = get_delchisq(llist[x])
        alphacrit_x = get_confints(alphalist[x], delchisqlist_x,
            delchisqcrit)

        # get most generous bounds on alphaNP
        alphamin_x = np.min(alphacrit_x)
        alphamax_x = np.max(alphacrit_x)

        alphamin.append(alphamin_x)
        alphamax.append(alphamax_x)

        #  multiplying back
        if showallowedfitpts:
            allowed_alphas_x = elem.dnorm * np.array([
                a for a in alphalist[x] if alphamin_x < a < alphamax_x])
            if plotabs:
                ax.scatter(elem.mphis[x] * np.ones(len(allowed_alphas_x)),
                    np.abs(allowed_alphas_x), color=fit_colour, s=1)

            else:
                ax.scatter(elem.mphis[x] * np.ones(len(allowed_alphas_x)),
                    allowed_alphas_x, color=fit_colour, s=1)

        if alphamin_x < 0:
            shrubberymin = np.max([alphamin_x, shrubberymin])
        if alphamax_x > 0:
            shrubberymax = np.min([np.max(alphacrit_x), shrubberymax])

    # fit_label = 'fit ' + str(nsigmas) + r'$\sigma$-excluded'

    maxabsalphas = np.max([np.abs(alphamin), np.abs(alphamax)], axis=0)

    nsteps = int(np.floor(len(elem.mphis) / 10))
    # mphi_bins = (np.min(elem.mphis)
    #     + np.arange(nsteps) * (np.max(elem.mphis) - np.min(elem.mphis)) / nsteps)

    bininds = np.array_split(np.arange(len(elem.mphis)), nsteps)

    bin_mphis = []
    bin_alphas = []

    for inds in bininds:
        binind = np.argmax(maxabsalphas[inds])
        bin_mphis.append((elem.mphis[inds])[binind])
        bin_alphas.append((maxabsalphas[inds])[binind])

    bin_alphas = elem.dnorm * np.array(bin_alphas)   # here multiplying back

    ax.plot(bin_mphis, bin_alphas, color=fit_colour)  # , label=fit_label)

    f = interp1d(bin_mphis, bin_alphas,
        kind='slinear', fill_value="extrapolate")
    fitinterpolpts = f(elem.mphis)
    ax.plot(elem.mphis, fitinterpolpts, color='darkgreen', linestyle='--')

    return ax, fitinterpolpts, shrubberymin, shrubberymax


def plot_mphi_alphaNP(elem, alphalist, llist, gkpdims, nmgkpdims, ndetsamples,
        nsigmas=2, plotabs=True,
        xlims=[None, None], ylims=[None, None],
        showallowedfitpts=False,
        showbestdetbounds=False, showalldetbounds=False):

    """
    Plot the most stringent nsigmas-bounds on both positive and negative
    alphaNP, derived using the Generalised King-plot formula of dimensions d
    listed in dims and save the output under plotname in the plots directory.
    If showall=True, all bounds GKP bounds of the appropriate dimensions are
    shown.

    """
    fig, ax = plt.subplots()

    shrubberymax = 1e17
    shrubberymin = -1e17

    # fit
    ###########################################################################

    ax, fitinterpolpts, shrubberymin, shrubberymax = plot_mphi_alphaNP_fit_bound(
        ax, elem, alphalist, llist,
        shrubberymin, shrubberymax,
        nsigmas=nsigmas, plotabs=plotabs,
        showallowedfitpts=showallowedfitpts)

    # determinant methods
    ###########################################################################

    minpos = np.array([])
    maxneg = np.array([])

    for d, dim in enumerate(gkpdims):
        ax, minpos, maxneg, shrubberymin, shrubberymax = \
            plot_mphi_alphaNP_det_bound(ax, elem, d, dim, ndetsamples, nsigmas,
            minpos, maxneg, shrubberymin, shrubberymax,
            gkp=True,
            showbestdetbounds=showbestdetbounds,
            showalldetbounds=showalldetbounds)

    for d, dim in enumerate(nmgkpdims):
        ax, minpos, maxneg, shrubberymin, shrubberymax = \
            plot_mphi_alphaNP_det_bound(ax, elem, d + len(gkpdims), dim,
            ndetsamples, nsigmas, minpos, maxneg, shrubberymin, shrubberymax,
            gkp=False,
            showbestdetbounds=showalldetbounds,
            showalldetbounds=showalldetbounds)

    # formatting + plotting combined det bound
    ###########################################################################
    xlabel=r"$m_\phi~$[eV]"

    if plotabs:
        plotname = 'mphi_abs_alphaNP'
        ylabel = r"$|\alpha_{\mathrm{NP}}/\alpha_{\mathrm{EM}}|$"
    else:
        plotname = 'mphi_alphaNP'
        ylabel = r"$\alpha_{\mathrm{NP}}/\alpha_{\mathrm{EM}}$"

    gkp_label = (('(' + ', '.join(str(gd) for gd in gkpdims) + ')-dim GKP') if
        len(gkpdims) > 0 else '')
    nmgkp_label = (('(' + ', '.join(str(nmd) for nmd in nmgkpdims)
        + ')-dim NMGKP') if len(nmgkpdims) > 0 else '')
    label_coupling = ' + ' if (gkp_label != '' and nmgkp_label != '') else ''

    det_label = gkp_label + label_coupling + nmgkp_label

    ax.plot(elem.mphis, minpos, color=det_colour)  # , label=det_label)
    ax.plot(elem.mphis, maxneg, color=det_colour)

    if ylims[0] is None:
        ymin = np.nanmin([np.nanmin(maxneg), np.nanmin(fitinterpolpts)])
    else:
        ymin = ylims[0]

    if ylims[1] is None:
        ymax = np.nanmax([np.nanmax(minpos), np.nanmax(fitinterpolpts)])
    else:
        ymax = ylims[1]

    ax.fill_between(elem.mphis, fitinterpolpts, ymax, color=fit_colour,
        alpha=.3, label='fit ' + str(nsigmas) + r'$\sigma$-excluded')
    ax.fill_between(elem.mphis, minpos, ymax, color=det_colour, alpha=.3,
        label=det_label + ' ' + str(nsigmas) + r'$\sigma$-excluded')
    ax.fill_between(elem.mphis, ymin, maxneg, color=det_colour, alpha=.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylabel(ylabel)
    ax.axhline(y=0, c='k')

    shrubbery = 10 ** np.floor(np.log10(np.min([shrubberymax, -shrubberymin])))

    ax.set_xscale('log')
    if plotabs:
        ax.set_yscale('log')
    else:
        ax.set_yscale('symlog', linthresh=shrubbery)

    ax.legend()

    plt.savefig(_plot_path + "/" + plotname + "_" + elem.id + "_"
        + str(len(alphalist[0])) + "_fit_samples.pdf")

    return 0
