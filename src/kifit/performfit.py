import os
from typing import List

import numpy as np

from scipy.interpolate import interp1d
from scipy.linalg import cholesky, cho_factor, cho_solve
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


np.random.seed(2)


def linfit(p, x):
    return p[0] * x + p[1]


def linfit_x(p, y):
    return (y - p[1]) / p[0]


# def blocking(experiments: List[float], nblocks: int):
#     """
#     Blocking method to compute the statistical uncertainty of the MC simulation.
#
#     Args:
#         experiments (List[float]): list of the experiments results.
#     """
#
#     if (len(experiments) % nblocks != 0):
#         raise ValueError(f"Number of experiments has to be a multiple of\
#             nblocks. Here {len(experiments)} is not multiple of {nblocks}.")
#
#     block_size = int(len(experiments) / nblocks)
#
#     ave, est, err = [], [], []
#
#     for b in range(nblocks):
#         block_data = experiments[b * block_size: (b + 1) * block_size]
#         ave.append(np.mean(block_data))
#         est.append(np.mean(ave))
#         err.append(np.std(est))
#
#     return est, err
#

def blocking_bounds(lbs: List[float], ubs: List[float], nblocks: int,
        plot_output: bool = False):
    """
    Blocking method to compute the statistical uncertainty of the confidence
    intervals.

    Args:
        lbs: list of lower bounds for the different experiments.
        ubs: crazy big Swiss bank
        nblocks: number of blocks used by the blocking method.

    """
    if plot_output:
        print("Là je vais dessiner une chiée de figures.")

    if len(lbs) != len(ubs):
        raise ValueError(f"Lists of lower and upper bounds passed to the \
        blocking method should be of equal length. \
            len(lbs)={len(lbs)}, len(ubs)={len(ubs)}")
    if (len(lbs) % nblocks != 0):
        raise ValueError(f"Number of experiments has to be a multiple of\
            nblocks. Here {len(lbs)} is not multiple of {nblocks}.")
    # print("lbs", lbs)
    # print("ubs", ubs)
    # assert (lbs <= ubs).all()

    block_size = int(len(lbs) / nblocks)

    # parametric bootstrap
    lb_min, ub_max, lb_val, ub_val, sig_lb, sig_ub = [], [], [], [], [], []

    for b in range(nblocks):
        lb_block = lbs[b * block_size: (b + 1) * block_size]
        ub_block = ubs[b * block_size: (b + 1) * block_size]

        lb_min.append(np.min(lb_block))
        ub_max.append(np.max(ub_block))

        lb_val.append(np.mean(lb_min))
        ub_val.append(np.mean(ub_max))

        sig_lb.append(np.std(lb_min))
        sig_ub.append(np.std(ub_max))

    sig_LB = sig_lb[-1]
    sig_UB = sig_ub[-1]

    LB = lb_val[-1] - sig_LB
    UB = ub_val[-1] + sig_UB

    if plot_output:

        from kifit.plotfit import blocking_plot

        blocking_plot(
            nblocks=nblocks,
            estimations=lb_val,
            uncertainties=sig_lb,
            label="Lower bound",
            filename="blocking_lb"
        )
        blocking_plot(
            nblocks=nblocks,
            estimations=ub_val,
            uncertainties=sig_ub,
            label="Upper bound",
            filename="blocking_ub"
        )
    return LB, UB, sig_LB, sig_UB


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

#
# def choLL(absd, covmat, lam=1e-7):
#     """
#     For a given sample of absd, with the covariance matrix covmat, compute the
#     log-likelihood using the Cholesky decomposition of covmat.
#
#     """
#     # print("lam", lam)
#     chol_covmat = cholesky(covmat + lam * np.eye(covmat.shape[0]), lower=True)
#
#     # L, lower = cho_factor(covmat + lam * np.eye(covmat.shape[0]),
#     #     lower=True)
#     #
#     # logdetsigd = 2 * np.sum(np.log(np.diag(L)))
#     # d_sigdinv_d = absd.dot(cho_solve((L, lower), absd))
#
#     logdetsigd = 2 * np.sum(np.log(np.diag(chol_covmat)))
#     d_sigdinv_d = absd.dot(cho_solve((chol_covmat, True), absd))
#
#     return 0.5 * (logdetsigd + d_sigdinv_d)
#
#     # A = np.linalg.solve(chol_covmat, absd)
#     # At = np.linalg.solve(chol_covmat.T, absd)
#
#     # ll = np.dot(A, A) / 2 + np.dot(At, At) / 2 + np.sum(np.log(np.diag(chol_covmat)))
#
#     # return ll
#


def choLL(absd, covmat, lam=0):
    """
    For a given sample of absd, with the covariance matrix covmat, compute the
    log-likelihood using the Cholesky decomposition of covmat.

    """
    chol_covmat, lower = cho_factor(covmat + lam * np.eye(covmat.shape[0]), lower=True)

    absd_covinv_absd = absd.dot(cho_solve((chol_covmat, lower), absd))

    logdet = 2 * np.sum(np.log(np.diag(chol_covmat)))

    return 0.5 * (logdet + absd_covinv_absd)

#
#
# def choLL(absd, covmat, lam=0):
#     # Regularize the covariance matrix and perform Cholesky decomposition
#     chol_covmat = cholesky(covmat + lam * np.identity(covmat.shape[0]), lower=True)
#
#     # Solve for A where chol_covmat * A = absd
#     A = np.linalg.solve(chol_covmat, absd)
#
#     # Compute the log-likelihood
#     ll = -0.5 * (np.dot(A, A) + 2 * np.sum(np.log(np.diag(chol_covmat))))
#
#

def get_llist(absdsamples, nsamples):
    """
    Get ll for list of absd-samples of length nsamples.

    """
    cov_absd = np.cov(np.array(absdsamples), rowvar=False)

    llist = []
    for s in range(nsamples):
        llist.append(choLL(absdsamples[s], cov_absd))

        # print("absd shape ", (absdsamples[s] @ absdsamples[s]).shape)

    return np.array(llist)


def generate_element_sample(elem, nsamples: int):
    """
    Generate ``nsamples`` of ``elem`` varying the input parameters according
    to the provided standard deviations.
    """
    parameter_samples = get_paramsamples(
        elem.means_input_params, elem.stdevs_input_params, nsamples
    )
    return parameter_samples


def generate_alphaNP_sample(elem, nsamples: int, search_mode: str = "random",
        lb: float = None, ub: float = None):
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
        if lb is None or ub is None:
            alphaNP_samples = np.linspace(
                elem.alphaNP_init - elem.sig_alphaNP_init,
                elem.alphaNP_init + elem.sig_alphaNP_init,
                nsamples
            )
        else:
            alphaNP_samples = np.linspace(
                lb,
                ub,
                nsamples
            )

    return np.sort(alphaNP_samples)


def get_new_alphaNP_interval(
        alphalist,
        llist,
        scalefactor: float = .1):
    print("may ll   ", max(llist))

    smalll_inds = np.argsort(llist)[: int(len(llist) * scalefactor)]

    # print(f"taking {int(len(llist) * scalefactor)} points to next level")

    small_alphas = alphalist[smalll_inds]
    smallls = llist[smalll_inds]
    print("max small", max(smallls))

    # sorted_alpha_inds = np.argsort(small_alphas)
    # small_alphas = small_alphas[sorted_alpha_inds]
    # smallls = smallls[sorted_alpha_inds]

    lb_best_alpha = np.percentile(small_alphas, 16)
    ub_best_alpha = np.percentile(small_alphas, 84)

    best_alpha = np.median(small_alphas)

    print("lb_best_alpha      ", lb_best_alpha)
    print("ub_best_alpha      ", ub_best_alpha)
    print("best_alpha", best_alpha)

    return (small_alphas, smallls,
        best_alpha, lb_best_alpha, ub_best_alpha)


def update_alphaNP_for_next_iteration(
        elem,
        new_alphas,
        new_lls,
        alphalist,
        llist,
        scalefactor: float = .1):  # ,
    # small_alpha_fraction: float = .3):
    """
    Compute sig_alphaNP for next iteration.

    """
    new_alpha = np.average(new_alphas)
    std_new_alphas = np.std(new_alphas)
    std_new_lls = np.std(new_lls)

    # lb_best_alphas = np.percentile(new_alphas, 45)
    # ub_best_alphas = np.percentile(new_alphas, 55)
    #
    # best_alpha_inds = np.argwhere(
    #     (lb_best_alphas < alphalist) & (alphalist < ub_best_alphas)).flatten()

    # best_alpha_interval_ll = llist[best_alpha_inds]
    # Delta_new_ll = np.abs(
    #     max(best_alpha_interval_ll) - min(best_alpha_interval_ll))
    Delta_new_ll = 0
    #
    # sig_new_alpha = scalefactor * min(
    sig_new_alpha = np.min([  # average
        np.abs(max(new_alphas) - new_alpha),
        np.abs(min(new_alphas) - new_alpha)])

    elem.set_alphaNP_init(new_alpha, sig_new_alpha)

    # print(f"""New search interval: \
    #     [{elem.alphaNP_init - elem.sig_alphaNP_init}, \
    #     {elem.alphaNP_init + elem.sig_alphaNP_init}]""")
    return std_new_lls, Delta_new_ll, new_alpha, std_new_alphas, sig_new_alpha


def get_bestalphaNP_and_bounds(
        bestalphaNPlist,
        confints,
        nblocks: int = 100,
        plot_output: bool = False):
    """
    Starting from the best alphaNP values, apply the blocking method to
    compute the best alphaNP value, its uncertainty, as well as the nsigma -
    upper and lower bounds on alphaNP.

    """
    confints = np.array(confints)

    best_alpha_pts = np.median(bestalphaNPlist)
    sig_alpha_pts = np.std(bestalphaNPlist)

    lowbounds_exps = confints.T[0]
    upperbounds_exps = confints.T[1]

    LB, UB, sig_LB, sig_UB = blocking_bounds(
        lowbounds_exps, upperbounds_exps, nblocks=nblocks,
        plot_output=plot_output)

    print(f"Final result: {best_alpha_pts} with bounds [{LB}, {UB}].")

    return (best_alpha_pts, sig_alpha_pts, LB, sig_LB, UB, sig_UB)


def compute_ll(elem, alphasamples, scalefactor: float = .1, elementsamples=None):
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

    if elementsamples is None:
        print("generating element sample")
        elemsamples = generate_element_sample(elem, nsamples)
    
    else:
        print("using same element samples")
        elemsamples = elementsamples

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


def get_delchisq(llist, minll=None):
    """
    Compute delta chi^2 from list of negative loglikelihoods, subtracting the
    minimum.

    """
    if minll is None:
        minll = min(llist)

    if len(llist) > 0:
        delchisqlist = 2 * (llist - minll)

        return delchisqlist

    else:
        raise ValueError(f"llist {llist} passed to get_delchisq is not a list.")


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


def equilibrate_interval(newalphas, newlls):

    sorted_alpha_inds = np.argsort(newalphas)
    newalphas = newalphas[sorted_alpha_inds]
    newlls = newlls[sorted_alpha_inds]
    print("len newlls before ", len(newlls))

    minll = min(newlls)
    min_llim = min(newlls[0], newlls[-1]) - minll
    max_llim = max(newlls[0], newlls[-1]) - minll

    llequill = min_llim / max_llim

    print("llequill", llequill)

    if llequill < .4:
        print("REGULATING UNBALANCED INTERVAL")
        limiting_ll = min(newlls[0], newlls[-1])
        # print("newlls     ", newlls)
        # print("min ll     ", min(newlls))
        # print("max ll     ", max(newlls))
        # print("limiting_ll", limiting_ll)
        reduced_ll_inds = np.where(newlls <= limiting_ll)
        newalphas = newalphas[reduced_ll_inds]
        newlls = newlls[reduced_ll_inds]
        print("len newlls after", len(newlls))
    return newalphas, newlls


def iterative_mc_search(
        elem,
        nsamples_search: int = 200,
        nexps: int = 1000,
        nsamples_exp: int = 1000,
        nsigmas: int = 2,
        nblocks: int = 10,
        sigalphainit: float = 1.,
        scalefactor: float = .1,  # 2e-1,
        # sig_new_alpha_fraction: float = 0.1,
        maxiter: int = 1000,
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

    alphas_exps = []
    lls_exps = []
    bestalphas_exps = []

    if mphivar is False:
        iterations = tqdm(range(maxiter))
    else:
        iterations = range(maxiter)

    for i in iterations:
        print()
        print(f"Iterative search step {i+1}")
        if i == 0:
            # 0: start with random search
            elem.set_alphaNP_init(0., sigalphainit)

            alphasamples = generate_alphaNP_sample(elem, nsamples_search,
                search_mode="random")

            Delta_new_ll = 0
            std_new_ll = 1

        # 1 -> -1: grid search
        elif ((i < maxiter - 1) and (Delta_new_ll < std_new_ll)):
            alphasamples = generate_alphaNP_sample(elem, nsamples_search,
                    search_mode="grid")

        else:
            print(f"BREAKING AT ITER: {i+1}, with maxiter: {maxiter}")
            break

        # compute ll, parabola and possible new interval
        alphas, lls = compute_ll(elem, alphasamples)

        llmin_10 = np.percentile(lls, 10)

        if plot_output:
            delchisqlist = get_delchisq(lls)
            plot_mc_output(alphas, delchisqlist,
                plotname=f"{i + 1}", llmin=llmin_10)

        (
            new_alphas, new_lls,
            new_alpha, new_lb_alpha, new_ub_alpha
        ) = get_new_alphaNP_interval(alphas, lls, scalefactor=scalefactor)

        # test new interval, if good, update
        # print("max(new_alphas)    ", max(new_alphas))
        # new_alphas, new_lls = equilibrate_interval(new_alphas, new_lls)
        # print("max(new_alphas_eq) ", max(new_alphas))

        sf = scalefactor
        it = 1

        while sf < 1 and it < maxiter:

            new_alphas, new_lls = equilibrate_interval(new_alphas, new_lls)

            window_width = max(new_alphas) - min(new_alphas)

            print("min(window)     ", min(new_alphas) + window_width / 4)
            print("new alpha       ", new_alpha)
            print("max(window)     ", min(new_alphas) + 3 / 4 * window_width)

            if (
                    (min(new_alphas) + window_width / 4 < new_alpha)
                    and (new_alpha < (
                        min(new_alphas) + 3 / 4 * window_width))
            ):

                (
                    std_new_ll, Delta_new_ll, _, _, _
                ) = update_alphaNP_for_next_iteration(
                    elem, new_alphas, new_lls, alphas, lls
                )

                sf = 1
                # print("Delta_new_ll", Delta_new_ll)
                # print("std_new_ll", std_new_ll)

            else:
                # print("adjusting scalefactor")
                # it += 1
                # sf += it**2 * scalefactor**it
                # print("sf ", sf)
                # (
                #     new_alphas, new_lls,
                #     new_alpha_parabola, new_lb_alpha, new_ub_alpha
                # ) = get_new_alphaNP_interval(alphas, lls,
                #     scalefactor=sf)
                # it += 1
                new_alphas = alphas
                new_lls = lls

                print(f"{i} interval not updated")
                break

            # attempt to jump out of window

    # -1: final round: perform parabolic fits and use blocking method to

    allalphasamples = generate_alphaNP_sample(elem, nexps * nsamples_exp,
        search_mode="grid")

    # shuffle the sample
    np.random.shuffle(allalphasamples)

    for exp in range(nexps):
        # print("exp", exp)
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

        if plot_output:
            llmin_10 = np.percentile(lls, 10)
            delchisqlist = get_delchisq(lls)
            plot_mc_output(alphas, delchisqlist,
                plotname=f"m1_exp_{exp}", llmin=llmin_10)

    minll_10 = np.percentile(np.array(lls_exps).flatten(), 10)

    delchisqs_exps = []

    for s in range(nexps):
        delchisqs = get_delchisq(lls_exps[s], minll_10)
        delchisqs_exps.append(delchisqs)

    delchisqcrit = get_delchisq_crit(nsigmas=nsigmas, dof=1)

    confints_exps = np.array([get_confint(alphas_exps[s], delchisqs_exps[s],
        delchisqcrit) for s in range(nexps)])

    (best_alpha_pts, sig_alpha_pts,
        LB, sig_LB, UB, sig_UB) = \
        get_bestalphaNP_and_bounds(bestalphas_exps,
            confints_exps, nblocks=nblocks)

    elem.set_alphaNP_init(best_alpha_pts, sig_alpha_pts)

    if plot_output:
        plot_final_mc_output(elem, alphas_exps, delchisqs_exps,
            delchisqcrit,
            bestalphapt=best_alpha_pts, sigbestalphapt=sig_alpha_pts,
            lb=LB, siglb=sig_LB, ub=UB, sigub=sig_UB,
            nsigmas=nsigmas, xind=xind)

    return [
        np.array(alphas_exps), np.array(delchisqs_exps),
        delchisqcrit,
        best_alpha_pts, sig_alpha_pts,
        LB, sig_LB, UB, sig_UB,
        xind]  # * elem.dnorm


def sample_alphaNP_fit(
        elem,
        nsamples_search,
        nexps: int = 1000,
        nsamples_exp: int = 1000,
        nsigmas: int = 2,
        nblocks: int = 10,
        scalefactor: float = .1,
        sigalphainit: float = 1.,
        # sig_new_alpha_fraction: float = 0.1,
        maxiter: int = 1000,
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
            sigalphainit=sigalphainit,
            plot_output=plot_output,
            # sig_new_alpha_fraction=sig_new_alpha_fraction,
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
