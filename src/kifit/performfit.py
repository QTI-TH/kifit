import os
from typing import List

import numpy as np

from scipy.interpolate import interp1d
from scipy.linalg import cholesky
from scipy.odr import ODR, Model, RealData
from scipy.special import binom
from scipy.stats import chi2, linregress, multivariate_normal
from scipy.optimize import curve_fit

from tqdm import tqdm


_output_data_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data")
)

if not os.path.exists(_output_data_path):
    os.makedirs(_output_data_path)


np.random.seed(1)


def linfit(p, x):
    return p[0] * x + p[1]


def linfit_x(p, y):
    return (y - p[1]) / p[0]


def parabola(x, a, b, c):
    return a * x**2 + b * x + c


def parabola_alphaNPmin(popt):
    return -popt[1] / (2 * popt[0])


def parabola_llmin(popt):
    return popt[2] - popt[1]**2 / (4 * popt[0])


def blocking(experiments: List[float], nblocks: int):
    """
    Blocking method to compute the statistical uncertainty of the MC simulation.

    Args:
        experiments (List[float]): list of the experiments results.
    """

    if (len(experiments) % nblocks != 0):
        raise ValueError(f"Number of experiments has to be a multiple of\
            nblocks. Here {len(experiments)} is not multiple of {nblocks}.")

    block_size = int(len(experiments) / nblocks)

    ave, est, err = [], [], []

    for b in range(nblocks):
        block_data = experiments[b * block_size: (b + 1) * block_size]
        ave.append(np.mean(block_data))
        est.append(np.mean(ave))
        err.append(np.std(est))

    return est, err


def get_odr_residuals(p, x, y, sx, sy):

    v = 1 / np.sqrt(1 + p[0] ** 2) * np.array([-p[0], 1])
    z = np.array([x, y]).T
    sz = np.array([np.diag([sx[i] ** 2, sy[i] ** 2]) for i in range(len(x))])

    residuals = np.array([v @ z[i] - p[1] * v[1] for i in range(len(x))])
    sigresiduals = np.sqrt(np.array([v @ sz[i] @ v for i in range(len(x))]))

    return residuals, sigresiduals


def perform_linreg(isotopeshiftdata, reftrans_index: int = 0):
    """
    Perform linear regression.

    Args:
        data (normalised isotope shifts: rows=isotope pairs, columns=trans.)
        reference_transition_index (default: first transition)

    Returns:
        slopes, intercepts, Kperp, phi

    """

    x = isotopeshiftdata.T[reftrans_index]
    y = np.delete(isotopeshiftdata, reftrans_index, axis=1)

    betas = []
    sig_betas = []

    for i in range(y.shape[1]):
        res = linregress(x, y.T[i])
        betas.append([res.slope, res.intercept])
        sig_betas.append([res.stderr, res.intercept_stderr])

    betas = np.array(betas)
    sig_betas = np.array(sig_betas)

    ph1s = np.arctan(betas.T[0])
    sig_ph1s = np.array(
        [sig_betas[j, 0] / (1 + betas[j, 0]) for j in range(len(betas))]
    )

    kperp1s = betas.T[1] * np.cos(ph1s)
    sig_kperp1s = np.sqrt(
        (sig_betas.T[1] * np.cos(ph1s)) ** 2
        + (betas.T[1] * sig_ph1s * np.sin(ph1s)) ** 2
    )

    return (betas, sig_betas, kperp1s, ph1s, sig_kperp1s, sig_ph1s)


def perform_odr(isotopeshiftdata, sigisotopeshiftdata, reftrans_index: int = 0):
    """
    Perform separate orthogonal distance regression for each transition pair.

    Args:
        data (normalised isotope shifts: rows=isotope pairs, columns=trans.)
        reftrans_index (default: first transition)

    Returns:
        slopes, intercepts, kperp1, ph1, sig_kperp1, sig_ph1

    """
    lin_model = Model(linfit)

    x = isotopeshiftdata.T[reftrans_index]
    y = np.delete(isotopeshiftdata, reftrans_index, axis=1)

    sigx = sigisotopeshiftdata.T[reftrans_index]
    sigy = np.delete(sigisotopeshiftdata, reftrans_index, axis=1)

    betas = []
    sig_betas = []

    for i in range(y.shape[1]):
        data = RealData(x, y.T[i], sx=sigx, sy=sigy.T[i])
        # results = linregress(x, y.T[i])
        # beta_init = [results.slope, results.intercept]
        beta_init = np.polyfit(x, y.T[i], 1)
        odr = ODR(data, lin_model, beta0=beta_init)
        out = odr.run()
        betas.append(out.beta)
        sig_betas.append(out.sd_beta)

    betas = np.array(betas)
    sig_betas = np.array(sig_betas)

    ph1s = np.arctan(betas.T[0])
    sig_ph1s = np.arctan(sig_betas.T[0])

    kperp1s = betas.T[1] * np.cos(ph1s)
    sig_kperp1s = np.sqrt(
        (sig_betas.T[1] * np.cos(ph1s)) ** 2
        + (betas.T[1] * sig_ph1s * np.sin(ph1s)) ** 2
    )

    return (betas, sig_betas, kperp1s, ph1s, sig_kperp1s, sig_ph1s)


def get_paramsamples(means, stdevs, nsamples):
    """
    Get nsamples samples of the parameters described by means and stdevs.

    """
    return multivariate_normal.rvs(means, np.diag(stdevs**2), size=nsamples)


def print_progress(s, nsamples):
    if s % (nsamples // 100) == 0:
        prog = np.round(s / nsamples * 100, 1)
        print("Progress", prog, "%")
    return 0


def choLL(absd, covmat, lam=0):
    """
    For a given sample of absd, with the covariance matrix covmat, compute the
    log-likelihood using the Cholesky decomposition of covmat.

    """
    chol_covmat = cholesky(covmat + lam * np.identity(covmat.shape[0]), lower=True)

    A = np.linalg.solve(chol_covmat, absd)
    At = np.linalg.solve(chol_covmat.T, absd)

    ll = np.dot(A, A) / 2 + np.dot(At, At) / 2 + np.sum(np.log(np.diag(chol_covmat)))

    return ll


def get_llist(absdsamples, nsamples):
    """
    Get ll for list of absd-samples of length nsamples.

    """
    cov_absd = np.cov(np.array(absdsamples), rowvar=False)

    llist = []
    for s in range(nsamples):
        llist.append(choLL(absdsamples[s], cov_absd))

    return np.array(llist)


def generate_element_sample(elem, nsamples: int):
    """
    Generate ``nsamples`` of ``elem`` varying the input parameters according
    to the provided standard deviations.
    """
    parameters_samples = get_paramsamples(
        elem.means_input_params, elem.stdevs_input_params, nsamples
    )
    return parameters_samples


def generate_alphaNP_sample(elem, nsamples: int, search_mode: str = "random"):
    """
    Generate ``nsamples`` of alphaNP according to the initial conditions
    provided by the ``elem`` data. The sample can be generated either randomly
    or by use of a grid.

    """
    if search_mode == "random":
        alphaNP_samples = np.random.normal(
            elem.alphaNP_init, elem.sig_alphaNP_init, nsamples
        )
    elif search_mode == "grid":
        alphaNP_samples = np.linspace(
            elem.alphaNP_init - elem.sig_alphaNP_init,
            elem.alphaNP_init + elem.sig_alphaNP_init,
            nsamples
        )
    return np.sort(alphaNP_samples)


def update_alphaNP_for_next_iteration(
        elem,
        alphalist,
        llist,
        scalefactor: float = .3,
        small_alpha_fraction: float = .1,
    ):
    """
    Compute sig_alphaNP for next iteration.

    """
    nsamples = len(alphalist)
    smalll = np.argsort(llist)

    small_alphas = np.array([alphalist[ll] for ll in smalll[: int(nsamples * small_alpha_fraction)]])

    new_alpha = np.mean(small_alphas)

    std_new_alpha = np.std(small_alphas)

    sig_new_alpha = scalefactor * min(
        np.abs(max(alphalist) - new_alpha),
        np.abs(min(alphalist) - new_alpha))

    elem.set_alphaNP_init(new_alpha, sig_new_alpha)

    # print(f"""New search interval: \
    #     [{elem.alphaNP_init - elem.sig_alphaNP_init}, \
    #     {elem.alphaNP_init + elem.sig_alphaNP_init}]""")

    return new_alpha, std_new_alpha, sig_new_alpha


def get_bestalphaNP_and_bounds(
        bestalphaNPlist,
        optparams,
        confints,
        nblocks: int = 100,
        draw_output: bool = True):
    """
    Starting from a list of parabola parameters, apply the blocking method to
    compute the best alphaNP value, its uncertainty, as well as the nsigma -
    upper and lower bounds on alphaNP.

    """
    optparams = np.array(optparams)
    confints = np.array(confints)

    reconstruced_alphas = - optparams.T[1] / (2 * optparams.T[0])
    best_alpha_parabola = np.mean(reconstruced_alphas)
    sig_alpha_parabola = np.std(reconstruced_alphas)

    best_alpha_pts = np.mean(bestalphaNPlist)
    sig_alpha_pts = np.std(bestalphaNPlist)

    lowbounds_list = confints.T[0]
    upbounds_list = confints.T[1]

    iterative_lb, iterative_lb_err = blocking(lowbounds_list, nblocks=nblocks)
    iterative_ub, iterative_ub_err = blocking(upbounds_list, nblocks=nblocks)

    print(iterative_lb_err)
    print(iterative_ub_err)

    lb = iterative_lb[-1]
    ub = iterative_ub[-1]
    sig_LB = iterative_lb_err[-1]
    sig_UB = iterative_ub_err[-1]

    LB = lb or -1e-17
    UB = ub or 1e-17

    if draw_output:
        from kifit.plotfit import blocking_plot
        blocking_plot(
            nblocks=nblocks,
            estimations=iterative_lb,
            errors=iterative_lb_err,
            label="Lower bound",
            filename="blocking_lb"
        )
        blocking_plot(
            nblocks=nblocks,
            estimations=iterative_ub,
            errors=iterative_ub_err,
            label="Upper bound",
            filename="blocking_ub"
        )

    return (best_alpha_parabola, sig_alpha_parabola,
        best_alpha_pts, sig_alpha_pts, LB, sig_LB, UB, sig_UB)


def compute_ll(elem, alphasamples, scalefactor: float = 3e-1):
    """
    Generate alphaNP list for element ``elem`` according to ``parameters_samples``.

    Args:
        elem (Elem): target element.
        nsamples (int): number of samples.
        mphivar (bool): if ``True``, this procedure is repeated for all
            X-coefficients provided for elem.
        save_sample (bool): if ``True``, the parameters and alphaNP samples are saved.

    Return:
        List[float], List[float]: alphaNP samples and list of associated log likelihood.
    """
    nsamples = len(alphasamples)

    alphasamples = np.sort(alphasamples)

    elemsamples = generate_element_sample(elem, nsamples)
    fitparamsamples = np.tensordot(np.ones(nsamples), elem.means_fit_params, axes=0)

    fitparamsamples[:, -1] = alphasamples

    absdsamples = []

    for s in range(nsamples):
        elem._update_elem_params(elemsamples[s])
        elem._update_fit_params(fitparamsamples[s])
        absdsamples.append(elem.absd)

    llist = get_llist(np.array(absdsamples), nsamples)

    # return elem.dnorm * np.array(alphalist), elem.dnorm * np.array(llist)
    return np.array(alphasamples), np.array(llist)


def parabolic_fit(elem, alphalist, llist, plotfit=False, plotname=None):
    """
    Perform parabolic fit to alphaNP and loglikelihood values. No messing around
    with the minimum.

    Returns: parabola parameters.

    """
    ll = llist[0]
    al = alphalist[0]
    lu = llist[-1]
    au = alphalist[-1]
    lm = min(llist)
    am = alphalist[np.argmin(llist)]

    if am != al and am != au and al != au:
        a0 = (am * (ll - lu) + al * (lu - lm) + au * (lm - ll)) / (
            (al - am) * (al - au) * (am - au))

        b0 = (am**2 * (lu - ll) + au**2 * (ll - lm) + al**2 * (lm - lu)) / (
            (al - am) * (al - au) * (am - au))

        c0 = (am * (am - au) * au * ll + al * (al - am) * am * lu
            + al * au * (au - al) * lm) / ((al - am) * (al - au) * (am - au))
    else:
        if am == al or am == au:
            a0 = (au * ll - al * lu) / (al * (al - au) * au)
            b0 = (al**2 * lu - au**2 * ll) / (al * (al - au) * au)
            c0 = 0
        elif al == au:
            a0 = (lm - lu) / (am - au)**2
            b0 = 2 * au * (lu - lm) / (am - au)**2
            c0 = (am**2 * lu - 2 * am * au * lu + au**2 * lm) / (am - au)**2

    # print("a0, b0, c0")
    # print([a0, b0, c0])

    popt, _ = curve_fit(
        parabola,
        alphalist,
        llist,
        p0=[a0, b0, c0])
    
    if plotfit:
        from kifit.plotfit import plot_parabolic_fit
        plot_parabolic_fit(alphalist, llist, popt, plotname=plotname)

    return popt


def get_delchisq_popt(popt, minll):
    if len(popt)==3:
        newpopt = np.array([
            2 * popt[0], 2 * popt[1], 2 * (popt[2] - minll)])
        # newpopt = np.array([
        #     2 * popt[0], 2 * popt[1], popt[1]**2 / (2 * popt[0]) + 2 * minll])

        return newpopt


def get_delchisq(llist, minll=None, popt=[]):
    """
    Compute delta chi^2 from list of negative loglikelihoods, subtracting the
    minimum.

    """
    if minll is None:
        minll = min(llist)

    if len(llist) > 0:
        delchisqlist = 2 * (llist - min(llist))
    else:
        raise ValueError(f"llist {llist} passed to get_delchisq is not a list.")

    if len(popt)==3:
        newpopt = get_delchisq_popt(popt, minll)

        return delchisqlist, newpopt

    else:
        return delchisqlist


def get_delchisq_crit(nsigmas=2, dof=1):
    """
    Get chi^2 level associated to nsigmas for ``dof`` degrees of freedom.

    """

    conf_level = chi2.cdf(nsigmas**2, 1)

    return chi2.ppf(conf_level, dof)


def get_confint(alphas, delchisqs, delchisqcrit):
    """
    Get nsigmas-confidence intervals.

    Returns:
    delchisq_crit: Delta chi^2 value associated to nsigmas.
    paramlist[pos]: parameter values with Delta chi^2 values in the vicinity of
    delchisq_crit

    """
    pos = np.argwhere(delchisqs < delchisqcrit).flatten()

    if len(pos) > 2:
        return np.array([alphas[int(min(pos))], alphas[int(max(pos))]])
    else:
        return np.array([np.nan, np.nan])


def iterative_mc_search(
        elem,
        nsamples_search: int = 200,
        nexps: int = 1000,
        nsamples_exp: int = 1000,
        nsigmas: int = 2,
        nblocks: int = 10,
        sigalphainit=1e-7,
        scalefactor: float = 3e-1,
        sig_new_alpha_fraction: float = 0.25,
        maxiter: int = 3,
        plot_output: bool = False,
        xind=0,
        mphivar: bool = False):
    """
    Perform iterative search for best alphaNP value and the standard deviation.

    Args:
        elem (Elem): target element.
        nsamples_search (int): number of samples.
        nsigmas (int): confidence level in standard deviations for which the
            upper and lower bounds are computed.
        nblocks (int): number of blocks used for the determination of the mean
            and standard deviation in the last iteration.
        scalefactor (float): factor used to rescale the search interval.
        maxiter (int): maximum number of iterations spent within the iterative
            search.

    Return:
        mean_best_alpha: best alphaNP value, found by averaging over all blocks
        sig_best_alpha: standard deviation of the best_alphas over all blocks
        LB: lower nsigma-bound on alphaNP
        sig_LB: uncertainty on LB
        UB: upper nsigma-bound on alphaNP
        sig_UB: uncertainty on UB

    """

    from kifit.plotfit import plot_mc_output, plot_final_mc_output

    optparams_exps = []
    alphas_exps = []
    lls_exps = []
    bestalphas_exps = []

    if mphivar is False:
        iterations = tqdm(range(maxiter))
    else:
        iterations = range(maxiter)

    for i in iterations:
        # print(f"Iterative search step {i+1}")
        if i == 0:
            # 0: start with random search
            elem.set_alphaNP_init(0., sigalphainit)

            alphasamples = generate_alphaNP_sample(elem, nsamples_search,
                search_mode="random")
            alphas, lls = compute_ll(elem, alphasamples)

            popt = parabolic_fit(elem, alphas, lls,
                plotfit=plot_output, plotname=str(i))

            _, std_new_alpha, sig_new_alpha = \
                update_alphaNP_for_next_iteration(elem, alphas, lls,
                    scalefactor=scalefactor)

            if plot_output:
                delchisqlist, newpopt = get_delchisq(lls, popt=popt)

                plot_mc_output(alphas, delchisqlist, newpopt, plotname=f"{i}")

        else:
            if (i < maxiter - 1) and (std_new_alpha < sig_new_alpha * sig_new_alpha_fraction):

                # 1-> -1: switch to grid search
                alphasamples = generate_alphaNP_sample(elem, nsamples_search,
                    search_mode="grid")
                alphas, lls = compute_ll(elem, alphasamples)

                popt = parabolic_fit(elem, alphas, lls,
                    plotfit=False, plotname=str(i))
                new_alpha, std_new_alpha, sig_new_alpha = \
                    update_alphaNP_for_next_iteration(elem, alphas, lls,
                        scalefactor=scalefactor)

                if plot_output:
                    delchisqlist, newpopt = get_delchisq(lls, popt=popt)
                    plot_mc_output(alphas, delchisqlist, newpopt, plotname=f"{i}")

            else:
                # -1: final round: perform parabolic fits and use blocking method to
                # determine best alphaNP and confidence intervals

                # generating a big number of data
                # we will perform nexps experiments, each of them
                # considering nsamples_exp data
                allalphasamples = generate_alphaNP_sample(elem, nexps * nsamples_exp,
                    search_mode="grid")

                # shuffling the sample
                np.random.shuffle(allalphasamples)

                for exp in range(nexps):
                    # collect data for a single experiment
                    alphasamples = allalphasamples[
                        exp * nsamples_exp: (exp + 1) * nsamples_exp]

                    # compute alphas and LLs for this experiment
                    alphas, lls = compute_ll(elem, alphasamples)

                    # save all alphas, lls and best alpha of the sample
                    # TODO: I think this is not necessary. We only need to save
                    #       the extremes of the interval
                    alphas_exps.append(alphas)
                    bestalphas_exps.append(alphas[np.argmin(lls)])
                    lls_exps.append(lls)

                    popt = parabolic_fit(elem, alphas, lls,
                        plotfit=False, plotname=f"{i}_exp_{exp}")
                    optparams_exps.append(popt)


                    if plot_output:
                        delchisqlist, newpopt = get_delchisq(lls, popt=popt)
                        plot_mc_output(alphas, delchisqlist, newpopt,
                            plotname=f"{i}_exp_{exp}")
                break

    # minll = np.min(lls_exps)
    #
    #         popt = parabolic_fit(elem, alphas, lls,
    #             plotfit=plot_output, plotname=str(i))
    #
    global_popt = parabolic_fit(elem, np.array(alphas_exps).flatten(),
        np.array(lls_exps).flatten())

    minll = parabola_llmin(global_popt)
    delchisq_optparams = get_delchisq_popt(global_popt, minll)

    delchisqs_exps = []
    delchisq_optparams_exps = []

    for s in range(nexps):
        delchisqs, params = get_delchisq(lls_exps[s], minll, popt=optparams_exps[s])
        delchisqs_exps.append(delchisqs)
        delchisq_optparams_exps.append(params)

    delchisqcrit = get_delchisq_crit(nsigmas=nsigmas, dof=1)

    confints_exps = np.array([get_confint(alphas_exps[s], delchisqs_exps[s],
        delchisqcrit) for s in range(nexps)])

    (best_alpha_parabola, sig_alpha_parabola, best_alpha_pts, sig_alpha_pts,
        LB, sig_LB, UB, sig_UB) = \
        get_bestalphaNP_and_bounds(bestalphas_exps, optparams_exps,
            confints_exps, nblocks=nblocks)

    elem.set_alphaNP_init(best_alpha_parabola, sig_alpha_parabola)

    if plot_output:
        plot_final_mc_output(elem, alphas_exps, delchisqs_exps,
            delchisq_optparams_exps, delchisq_optparams, delchisqcrit,
            bestalphaparabola=best_alpha_parabola,
            sigbestalphaparabola=sig_alpha_parabola,
            bestalphapt=best_alpha_pts, sigbestalphapt=sig_alpha_pts,
            lb=LB, siglb=sig_LB, ub=UB, sigub=sig_UB,
            nsigmas=nsigmas, xind=xind)

    return [
        np.array(alphas_exps), np.array(delchisqs_exps),
        np.array(delchisq_optparams_exps), np.array(delchisq_optparams),
        delchisqcrit,
        best_alpha_parabola, sig_alpha_parabola, best_alpha_pts, sig_alpha_pts,
        LB, sig_LB, UB, sig_UB,
        xind]  # * elem.dnorm


def sample_alphaNP_fit(
        elem,
        nsamples_search,
        nexps: int = 1000,
        nsamples_exp: int = 1000,
        nsigmas: int = 2,
        nblocks: int = 10,
        scalefactor: float = 3e-1,
        maxiter: int = 3,
        sig_new_alpha_fraction: float = 0.25,
        plot_output: bool = False,
        mphivar: bool = False,
        x0: int = 0):
    """
    Get a set of nsamples_search samples of elem by varying the masses and isotope
    shifts according to the means and standard deviations given in the input
    files, as well as alphaNP.

       m  ~ N(<m>,  sig[m])
       m' ~ N(<m'>, sig[m'])
       v  ~ N(<v>,  sig[v])

       alphaNP ~ N(0, sig[alphaNP_init]).

    If mphivar=True, this procedure is repeated for all X-coefficients provided
    for elem.

    """
    if mphivar:
        x_range = tqdm(range(len(elem.Xcoeff_data)))
    else:
        x_range = [x0]

    res_list = []

    for x in x_range:
        elem._update_Xcoeffs(x)

        res = iterative_mc_search(
            elem=elem,
            nsamples_search=nsamples_search,
            nexps=nexps,
            nsamples_exp=nsamples_exp,
            nsigmas=nsigmas,
            nblocks=nblocks,
            scalefactor=scalefactor,
            maxiter=maxiter,
            sig_new_alpha_fraction=sig_new_alpha_fraction,
            plot_output=plot_output,
            xind=x,
            mphivar=mphivar)

        res_list.append(res)

    mc_output = [res_list, nsigmas]

    # file_path = (_output_data_path + "/mc_output_" + elem.id + "_"
    #     + str(nsigmas) + "sigmas_" + str(nsamples_exp) + "_samples.pdf")
    # np.save(file_path, mc_output)

    return mc_output


# DETERMINANT METHODS

def sample_alphaNP_det(
    elem, dim, nsamples, mphivar=False, gkp=True, outputdataname="alphaNP_det",
    x0=0
):
    """
    Get a set of nsamples samples of alphaNP by varying the masses and isotope
    shifts according to the means and standard deviations given in the input
    files:

       m  ~ N(<m>,  sig[m])
       m' ~ N(<m'>, sig[m'])
       v  ~ N(<v>,  sig[v])

    For each of these samples and for all possible combinations of the data,
    compute alphaNP using the Generalised King Plot formula with

        (nisotopepairs, ntransitions) = (dim, dim-1).

    """
    print()
    print(
        """Using the %s-dimensional %sGeneralised King Plot to compute
    alphaNP for %s samples. mphi is %svaried."""
        % (dim, "" if gkp else "No-Mass ", nsamples, ("" if mphivar else "not "))
    )

    if gkp:
        method_tag = "GKP"
    else:
        method_tag = "NMGKP"

    file_path = (_output_data_path + "/" + outputdataname + "_" + elem.id + "_"
        + str(dim) + "-dim_" + method_tag + "_" + str(nsamples) + "_samples.pdf")

    if os.path.exists(file_path) and elem.id != "Ca_testdata":
        print()
        print("Loading alphaNP and sigalphaNP values from {}".format(file_path))
        print()

        # preallocate
        if gkp:
            vals = np.zeros(
                (len(elem.mphis), 2 * int(binom(elem.ntransitions, dim - 1)))
            )

        else:
            vals = np.zeros((len(elem.mphis), 2 * int(binom(elem.ntransitions, dim))))

        vals = np.loadtxt(file_path, delimiter=",")  # [x][2*perm]
        alphaNPs = vals[:, : int((vals.shape[1]) / 2)]
        sigalphaNPs = vals[:, int((vals.shape[1]) / 2):]

    else:
        print()
        print("""Initialising alphaNP""")
        print()
        elemparamsamples = get_paramsamples(
            elem.means_input_params, elem.stdevs_input_params, nsamples
        )

        voldatsamples = []
        vol1samples = []

        # nutilsamples = []   # add mass-normalised isotope shifts for cross-check

        for s in range(nsamples):
            # print_progress(s, nsamples)
            elem._update_elem_params(elemparamsamples[s])

            if gkp:
                alphaNPparts = elem.alphaNP_GKP_part(dim)
            else:
                alphaNPparts = elem.alphaNP_NMGKP_part(dim)  # this is new

            voldatsamples.append(alphaNPparts[0])
            vol1samples.append(alphaNPparts[1])
            if s == 0:
                xindlist = alphaNPparts[2]
            else:
                assert xindlist == alphaNPparts[2], (xindlist, alphaNPparts[2])

        # voldatsamples has the form [sample][alphaNP-permutation]
        # vol1samples has the form [sample][alphaNP-permutation][eps-term]

        # for each term, average over all samples.

        meanvoldat = np.average(np.array(voldatsamples), axis=0)  # [permutation]
        sigvoldat = np.std(np.array(voldatsamples), axis=0)

        meanvol1 = np.average(np.array(vol1samples), axis=0)  # [perm][eps-term]
        sigvol1 = np.std(np.array(vol1samples), axis=0)

        print()
        print("""Computing alphaNP""")
        print()
        if mphivar:
            x_range = range(len(elem.Xcoeff_data))
        else:
            x_range = [x0]

        # mphi_list = []
        alphaNPs = []  # alphaNP list for best alphaNP and all
        sigalphaNPs = []

        for x in x_range:
            if mphivar:
                elem._update_Xcoeffs(x)
                # mphi_list.append(elem.mphi)
                # print_progress(nsamples * x, nsamples * Nx)

            """ p: alphaNP-permutation index and xpinds: X-indices for sample p"""
            alphaNP_p_list = []
            sig_alphaNP_p_list = []
            for p, xpinds in enumerate(xindlist):
                meanvol1_p = np.array([elem.Xvec[xp] for xp in xpinds]) @ (meanvol1[p])
                sigvol1_p_sq = np.array([elem.Xvec[xp] ** 2 for xp in xpinds]) @ (
                    sigvol1[p] ** 2
                )

                alphaNP_p_list.append(meanvoldat[p] / meanvol1_p)
                sig_alphaNP_p_list.append(
                    (sigvoldat[p] / meanvol1_p) ** 2
                    + (meanvoldat[p] / meanvol1_p**2) ** 2 * sigvol1_p_sq
                )
            alphaNPs.append(alphaNP_p_list)
            sigalphaNPs.append(sig_alphaNP_p_list)

        alphaNPs = np.math.factorial(dim - 2) * np.array(alphaNPs)
        sigalphaNPs = np.math.factorial(dim - 2) * np.array(sigalphaNPs)

        np.savetxt(file_path, np.c_[alphaNPs, sigalphaNPs], delimiter=",")

    return alphaNPs, sigalphaNPs


def get_all_alphaNP_bounds(alphaNPs, sigalphaNPs, nsigmas=2):
    """
    Determine all bounds on alphaNP at the desired confidence level.

    """
    alphaNPs = np.array(alphaNPs)  # [x][perm]
    sigalphaNPs = np.array(sigalphaNPs)  # [x][perm]

    # minimal positive alphaNP
    ###############################
    # Note: For NP we only want an exclusion bound, hence we do not consider the
    # case of both upper and lower limits being positive / negative.

    alphaNP_UB = alphaNPs + nsigmas * sigalphaNPs
    alphaNP_LB = alphaNPs - nsigmas * sigalphaNPs

    positive_alphaNP_bounds = np.where(alphaNP_UB > 0, alphaNP_UB, np.nan)
    negative_alphaNP_bounds = np.where(alphaNP_LB < 0, alphaNP_LB, np.nan)

    return positive_alphaNP_bounds, negative_alphaNP_bounds


def get_minpos_maxneg_alphaNP_bounds(alphaNPs, sigalphaNPs, nsigmas=2):
    """
    Determine smallest positive and largest negative values for the bound on
    alphaNP at the desired confidence level.

    """
    alphaNP_UBs, alphaNP_LBs = get_all_alphaNP_bounds(
        alphaNPs, sigalphaNPs, nsigmas=nsigmas
    )

    minpos = np.nanmin(alphaNP_UBs, axis=1)
    maxneg = np.nanmax(alphaNP_LBs, axis=1)

    return minpos, maxneg
