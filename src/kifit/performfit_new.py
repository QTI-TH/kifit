import os
import json
import logging 
import datetime
from pathlib import Path
from typing import List
from itertools import (
    product, 
    combinations, 
    groupby
)

import numpy as np

from scipy.linalg import cho_factor, cho_solve
from scipy.odr import ODR, Model, RealData
from scipy.special import binom
from scipy.stats import chi2, linregress, multivariate_normal

from scipy.optimize import (
    minimize,
    dual_annealing,
    differential_evolution,
)

from kifit.loadelems import Elem

_output_data_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data")
)

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def generate_path(pathname:str):
    output_path = Path("results") / pathname
    if not output_path.exists():
        output_path.mkdir(parents=True)
    # path where to save plots
    plots_path = output_path / f"plots"
    if not plots_path.exists():
        plots_path.mkdir(parents=True)
    return output_path, plots_path

np.random.seed(27)

def linfit(p, x):
    return p[0] * x + p[1]


def linfit_x(p, y):
    return (y - p[1]) / p[0]


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


def spectraLL(absd, covmat, lam=0):
    """
    For a given sample of absd, with the covariance matrix covmat, compute the
    log-likelihood using the spectral decomposition of covmat.
    """
    covmat += lam * np.eye(covmat.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covmat)

    inv_eigenvalues = 1.0 / eigenvalues
    covinv = eigenvectors @ np.diag(inv_eigenvalues) @ eigenvectors.T

    absd_covinv_absd = absd.dot(covinv).dot(absd)

    logdet = np.sum(np.log(eigenvalues))

    return 0.5 * (logdet + absd_covinv_absd)


def get_llist(absdsamples, nelemsamples, cov_decomp_method="cholesky"):
    """
    For a fixed alphaNP value, get ll for the nelemsamples samples of the input
    parameters.

    """
    # compute covariance matrix for fixed alpha value
    cov_absd = np.cov(np.array(absdsamples), rowvar=False)

    if cov_decomp_method == "cholesky":
        LL = choLL
    elif cov_decomp_method == "spectral":
        LL = spectraLL

    llist = []
    for s in range(nelemsamples):
        llist.append(LL(absdsamples[s], cov_absd))

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
    lb_nans = np.argwhere(np.isnan(lowbounds_exps))
    upperbounds_exps = confints.T[1]
    ub_nans = np.argwhere(np.isnan(upperbounds_exps))

    np.alltrue(lb_nans == ub_nans)

    lowbounds_exps = lowbounds_exps[~np.isnan(lowbounds_exps)]
    upperbounds_exps = upperbounds_exps[~np.isnan(upperbounds_exps)]

    LB, UB, sig_LB, sig_UB = blocking_bounds(
        lowbounds_exps, upperbounds_exps, nblocks=nblocks,
        plot_output=plot_output)

    print(f"Final result: {best_alpha_pts} with bounds [{LB}, {UB}].")

    return (best_alpha_pts, sig_alpha_pts, LB, sig_LB, UB, sig_UB)


def compute_ll(elem, alphasamples, nelemsamples,
        elementsamples=None, cov_decomp_method="cholesky"):
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

    if elementsamples is None:
        # print("generating element sample")
        elemsamples = generate_element_sample(elem, nelemsamples)

    else:
        print("reusing element samples")
        elemsamples = elementsamples

    # sampling fit parameters
    nalphasamples = len(alphasamples)
    # same intercept and slope for all, only alphaNP is updated
    fitparamsamples = np.tensordot(np.ones(nalphasamples), elem.means_fit_params, axes=0)
    fitparamsamples[:, -1] = alphasamples

    alphalist = []
    llist = []

    for s in range(nalphasamples):
        absdsamples_alpha = []

        for t in range(nelemsamples):
            elem._update_elem_params(elemsamples[t])
            elem._update_fit_params(fitparamsamples[s])
            absdsamples_alpha.append(elem.absd)

        alphalist.append(np.ones(nelemsamples) * alphasamples[s])
        llist.append(get_llist(np.array(absdsamples_alpha), nelemsamples,
            cov_decomp_method))

    # return elem.dnorm * np.array(alphalist), elem.dnorm * np.array(llist)
    return np.array(alphalist).flatten(), np.array(llist).flatten()


def logL_alphaNP(alphaNP, elem_collection, elemsamples_collection):
    # elem = args[0]
    # elemsamples = args[1]

    for elem in elem_collection:
        fitparams = elem.means_fit_params
        fitparams[-1] = alphaNP
        elem._update_fit_params(fitparams)

    # take on the two lists length
    nelemsamples = len(elemsamples_collection[0])

    loss = np.zeros(nelemsamples)

    # loop over elements in the collection
    for i, elem in enumerate(elem_collection):
        # for each element compute LL independently
        absdsamples = []
        for s in range(nelemsamples):
            elem._update_elem_params(elemsamples_collection[i][s])
            absdsamples.append(elem.absd)
        lls = get_llist(np.array(absdsamples), nelemsamples)
        delchisq = get_delchisq(lls)
        loss += delchisq
    
    return np.percentile(loss, 10)


def minimise_logL_alphaNP(
        elem_collection, elemsamples_collection, alpha0, maxiter, opt_method, tol=1e-12
    ):

    if opt_method == "annealing":
        minlogL = dual_annealing(
            logL_alphaNP, 
            bounds=[(-1e-4, 1e-4)], 
            args=(elem_collection, elemsamples_collection), 
            maxiter=maxiter,
        )

    elif opt_method == "differential_evolution":
        minlogL = differential_evolution(
            logL_alphaNP, 
            bounds=[(-1e-4, 1e-4)], 
            args=(elem_collection, elemsamples_collection), 
            maxiter=maxiter,
        )

    else:
        minlogL = minimize(
            logL_alphaNP, 
            x0=0, 
            bounds=[(-1e-6, 1e-6)],
            args=(elem_collection, elemsamples_collection),
            method=opt_method, options={"maxiter": maxiter}, 
            tol=tol,
        )

    return minlogL


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

    # Best 1% of points was used to define minll.
    # Make sure there are more than 1% of points below delchisqcrit.

    if len(pos) > 2:

        if len(pos) < int(len(delchisqs) / 100):
            print(f"Npts in confidence interval:     {len(pos)}")
            print(f"Npts used for def. of min(logL): {int(len(delchisqs) / 100)}")
            print(" ==> Need to generate more points.")

        return np.array([alphas[int(min(pos))], alphas[int(max(pos))]])
    else:
        return np.array([np.nan, np.nan])


def determine_search_interval(
        elem_collection,
        nsearches,
        nelemsamples_search,
        alpha0,
        opt_method,
        maxiter,
        verbose,
    ):

    # sampling `nsamples` new elements for each element in the collections
    allelemsamples = []
    for elem in elem_collection:
        allelemsamples.append(
            generate_element_sample(
                elem,
                nsearches * nelemsamples_search
            )
        )

    best_alpha_list = []

    print("scipy minimisation")
    for search in tqdm(range(nsearches)):
        if verbose:
            logging.info(f"Iterative search {search + 1}/{nsearches}")
        
        elemsamples_collection = []
        for i in range(len(allelemsamples)):
            elemsamples_collection.append(
                allelemsamples[i][
                    search * nelemsamples_search: (search + 1) * nelemsamples_search
                ]
            )

        res_min = minimise_logL_alphaNP(
            elem_collection=elem_collection, 
            elemsamples_collection=elemsamples_collection,
            alpha0=alpha0, 
            maxiter=maxiter, 
            opt_method=opt_method
        )
        
        if res_min.success:
            best_alpha_list.append(res_min.x[0])

        # print("best alpha search " + str(search) + ": ", res_min.x[0])

    best_alpha = np.median(best_alpha_list)
    # print("median", best_alpha)
    # print("average", np.average(best_alpha_list))

    print("EXTREMES", max(best_alpha_list), min(best_alpha_list))

    sig_best_alpha = max(best_alpha_list) - min(best_alpha_list)

    # print("sig_best_alpha", sig_best_alpha)
    # print("std_best_alpha", np.std(best_alpha_list))

    elem.set_alphaNP_init(best_alpha, sig_best_alpha)

    return best_alpha, sig_best_alpha


def perform_experiments(
        elem,
        delchisqcrit,
        nexps,
        nelemsamples_exp,
        nalphasamples_exp,
        nblocks,
        nsigmas,
        plot_output,
        verbose,
        plots_path,
        xind=0,
    ):

    from kifit.plotfit import plot_mc_output, plot_final_mc_output

    allalphasamples = generate_alphaNP_sample(elem, nexps * nalphasamples_exp,
        search_mode="random")

    # shuffle the sample
    np.random.shuffle(allalphasamples)

    alphas_exps = []
    lls_exps = []
    bestalphas_exps = []

    delchisqs_exps = []

    for exp in range(nexps):
        if verbose:
            logging.info(f"Running experiment {exp+1}/{nexps}")
  
        # collect data for a single experiment
        alphasamples = allalphasamples[
            exp * nalphasamples_exp: (exp + 1) * nalphasamples_exp]

        # compute alphas and LLs for this experiment
        alphas, lls = compute_ll(elem, alphasamples, nelemsamples_exp)

        alphas_exps.append(alphas)
        bestalphas_exps.append(alphas[np.argmin(lls)])
        lls_exps.append(lls)

        minll_1 = np.percentile(lls, 10)
        delchisqlist = get_delchisq(lls, minll=minll_1)

        if plot_output:
            plot_mc_output(
                alphalist=alphas, 
                delchisqlist=delchisqlist,
                plotname=f"exp_{exp}", 
                minll=minll_1,
                plot_path=plots_path,
            )

        delchisqs_exps.append(delchisqlist)

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
            nsigmas=nsigmas, xind=xind, plot_path=plots_path)

    return [
        np.array(alphas_exps), np.array(delchisqs_exps),
        delchisqcrit,
        best_alpha_pts, sig_alpha_pts,
        LB, sig_LB, UB, sig_UB,
        xind]


def sample_alphaNP_fit(
        elem_collection: list[Elem],
        output_filename: str,
        nsearches: int = 10,
        nelemsamples_search: int = 100,
        nexps: int = 100,
        nelemsamples_exp: int = 1000,
        nalphasamples_exp: int = 1000,
        nblocks: int = 10,
        nsigmas: int = 2,
        maxiter: int = 1000,
        plot_output: bool = False,
        alpha0=0.,
        mphivar: bool = False,
        opt_method: str = "Powell",
        x0: int = 0,
        verbose: bool = True,
    ):
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

    _output_path, _plots_path = generate_path(output_filename) 

    # saving utils
    result_filenames = [
        "alpha_experiments",
        "delchisq_experiments",
        "delchisq_crit",
        "best_alpha_pts",
        "sig_alpha_pts",
        "LB", "sig_LB",
        "UB", "sig_UB",
        "x_ind",
    ]

    # check the Xcoeff are the same
    first_list = np.round(np.asarray(elem_collection[0].Xcoeff_data).T[0], decimals=7)
    for elem in elem_collection:
        new_list = np.round(np.asarray(elem.Xcoeff_data).T[0], decimals=7)
        if (new_list != first_list).any():
            raise ValueError(
                "Please prepare data with same mphi values for all the elements in the collection."
            )
    logging.info("All elements respect the initialization requirements.")


    if mphivar:
        x_range = range(len(elem_collection[0].Xcoeff_data))
    else:
        x_range = [x0]

    res_list = []

    delchisqcrit = get_delchisq_crit(nsigmas=nsigmas, dof=1)

    for x in x_range:
        # path where to save results for mass index x
        index_path = _output_path / f"x{x}"
        if not index_path.exists():
            index_path.mkdir(parents=True)
        
        for elem in elem_collection:
            elem._update_Xcoeffs(x)

        alpha_optimizer, sig_alpha_optimizer = determine_search_interval(
            elem_collection=elem_collection,
            nsearches=nsearches,
            nelemsamples_search=nelemsamples_search,
            alpha0=alpha0,
            maxiter=maxiter,
            opt_method=opt_method,
            verbose=verbose,
        )

        res_exp = perform_experiments(
            elem=elem,
            delchisqcrit=delchisqcrit,
            nexps=nexps,
            nelemsamples_exp=nelemsamples_exp,
            nalphasamples_exp=nalphasamples_exp,
            nblocks=nblocks,
            nsigmas=nsigmas,
            plot_output=plot_output,
            xind=x,
            verbose=verbose,
            plots_path=_plots_path,
        )
        
        for i, res in enumerate(res_exp):
            np.save(arr=res, file=index_path/result_filenames[i])

        res_list.append(res_exp)

    output_results = {
        "nsigmas": nsigmas,
        "optimizer": opt_method,
        "maxiter": maxiter,
        "nsearches": nsearches,
        "nexps": nexps,
        "mphivar": mphivar,
        "nelemsamples_exp": nelemsamples_exp,
        "nalphasamples_exp": nalphasamples_exp,
        "nelemsamples_search": nelemsamples_search,
        "alpha_optimizer": alpha_optimizer,
        "sig_alpha_optimizer": sig_alpha_optimizer,
    }

    dict_path = _output_path / "optimization_config.json"
    dict_path.write_text(json.dumps(output_results, indent=4))

    mc_output = [res_list, nsigmas]

    return mc_output


# DETERMINANT METHODS

def sample_alphaNP_det(
    elem, 
    output_filename,
    dim, 
    nsamples,
    mphivar=False, 
    gkp=True, 
    outputdataname="alphaNP_det",
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
    alphaNP for %s samples. mphi is %svaried. %s"""
        % (
            dim,
            "" if gkp else "No-Mass ",
            nsamples,
            ("" if mphivar else "not "),
            ("" if mphivar
                else "(mphi=" + str(elem.mphi) + ", x0=" + str(elem.x) + ")")))

    if gkp:
        method_tag = "GKP"
    else:
        method_tag = "NMGKP"
    
    mphi_tag = ("mphivar" if mphivar else ("x" + str(int(x0)))) 

    _output_path, _ = generate_path(output_filename)

    file_path_tail = f"{outputdataname}_{elem.id}_{str(dim)}dim_{method_tag}_{str(nsamples)}_samples_{mphi_tag}.txt"
   
    file_path = (_output_data_path + "/" + file_path_tail)
    sig_file_path = (_output_data_path + "/sig" + file_path_tail)

    if (os.path.exists(file_path) and os.path.exists(sig_file_path)
            and elem.id != "Ca_testdata"):
        print()
        print("Loading alphaNP and sigalphaNP values from {}".format(file_path))
        print()

        if gkp:
            lenp = len(list(
                product(
                    combinations(elem.range_a, dim),
                    combinations(elem.range_i, dim - 1))))

        else:
            lenp = len(list(
                product(
                    combinations(elem.range_a, dim),
                    combinations(elem.range_i, dim))))

        if mphivar:
            lenx = len(elem.mphis)
        else:
            lenx = 1

        alphaNPs = np.loadtxt(file_path, delimiter=",").reshape(lenx, lenp)
        sigalphaNPs = np.loadtxt(sig_file_path, delimiter=",").reshape(lenx, lenp)
        # [x][perm]

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

        np.savetxt(file_path, alphaNPs, delimiter=",")
        np.savetxt(sig_file_path, sigalphaNPs, delimiter=",")

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

    all vectors have dimensions [x][perm]

    """
    alphaNP_UBs, alphaNP_LBs = get_all_alphaNP_bounds(
        alphaNPs, sigalphaNPs, nsigmas=nsigmas
    )

    minpos = np.nanmin(alphaNP_UBs, axis=1)
    maxneg = np.nanmax(alphaNP_LBs, axis=1)

    return minpos, maxneg
