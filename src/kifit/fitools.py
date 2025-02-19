import logging
from typing import List

import numpy as np

from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2, multivariate_normal

from scipy.optimize import (
    minimize,
    dual_annealing,
    differential_evolution,
)


# keys for saving / reading data
##############################################################################

fit_keys = [
    "alpha_search",
    "sig_alpha_search",
    "alphas_exp",
    "delchisqs_exp",
    "best_alpha",
    "sig_best_alpha",
    "LB", "sig_LB",
    "UB", "sig_UB",
    "nsigmas",
    "x_ind"
]


# generate samples
##############################################################################

def generate_paramsamples(means, stdevs, nsamples):
    """
    Generate ``nsamples`` by sampling the multivariate normal distribution
    specified by by means and stdevs.

    """
    return multivariate_normal.rvs(means, np.diag(stdevs**2), size=nsamples)


def generate_elemsamples(elem, nsamples: int):
    """
    Generate ``nsamples`` of the input parameters associated to ``elem`` from
    the normal distributions defined by the means and standard deviations
    provided by the corresponding data files:


       m  ~ N(<m>,  sig[m])   for the nuclear masses
       v  ~ N(<v>,  sig[v])   for the transition frequencies


    """
    inputparamsamples = generate_paramsamples(
        elem.means_input_params, elem.stdevs_input_params, nsamples
    )

    # sampling fit params
    samples_per_transition = []
    for kp1, ph1, cov in zip(elem.kp1_init, elem.ph1_init, elem.cov_kperp1_ph1):
        samples_per_transition.append(
            np.random.multivariate_normal([kp1, ph1], cov, size=nsamples))
    fitparamsamples = []
    for i in range(nsamples):
        fitparams = []
        for j in range(len(elem.kp1_init)):
            fitparams.append(samples_per_transition[j][i][0])
        for j in range(len(elem.kp1_init)):
            fitparams.append(samples_per_transition[j][i][1])
        fitparamsamples.append(fitparams)

    return inputparamsamples, fitparamsamples


def objective(trial, elem_collection, nelemsamples, min_percentile,
        pos_alphaNP=True):

    alphaNP = trial.suggest_float('alphaNP', 1e-16, 1, log=True)

    if pos_alphaNP:
        loss = logL_alphaNP(alphaNP, elem_collection, nelemsamples, min_percentile)

    else:
        loss = logL_alphaNP(-alphaNP, elem_collection, nelemsamples, min_percentile)

    return loss


def generate_alphaNP_samples(elem,
        nsamples: int,
        search_mode: str = "normal",
        lbmin: float = None, lbmax: float = None,
        ubmin: float = None, ubmax: float = None,
        logridfrac: float = 17):
    """
    Generate ``nsamples`` of alphaNP according to the initial conditions set in
    the instance ``elem`` of the Elem class. alphaNP is either sampled from a
    normal distribution or from a grid, both of which are defined via the
    initial conditions on alphaNP.

    """
    if search_mode == "normal":
        if lbmin is not None and ubmax is not None:
            sigalphainit = np.max([
                np.abs(elem.alphaNP_init - lbmin),
                np.abs(ubmax - elem.alphaNP_init)])
        elif lbmax is not None or ubmin is not None:
            logging.info(
                """Ignoring lbmax and ubmin provided for alphaNP.
                Sampling from normal distribution.""")
        else:
            sigalphainit = elem.sig_alphaNP_init
        alphaNP_samples = np.random.normal(
            elem.alphaNP_init, sigalphainit, size=nsamples
        )
    elif search_mode == "lingrid":
        if lbmin is None or ubmax is None:
            lb = elem.alphaNP_init - elem.sig_alphaNP_init
            ub = elem.alphaNP_init + elem.sig_alphaNP_init
        elif lbmax is not None or ubmin is not None:
            logging.info(
                """Ignoring lbmax and ubmin provided for alphaNP.
                Sampling from linear grid.""")
        alphaNP_samples = np.linspace(
            lb,
            ub,
            num=nsamples
        )
    elif search_mode == "globalogrid":
        posgrid = np.logspace(-15, -1, nsamples)
        alphaNP_samples = np.concatenate((- posgrid[::-1], posgrid))

    elif search_mode == "detlogrid":
        if lbmin is None or ubmax is None:
            lbmin = elem.alphaNP_init - elem.sig_alphaNP_init
            ubmax = elem.alphaNP_init + elem.sig_alphaNP_init

        if lbmin < 0 and ubmax < 0:
            alphaNP_samples = -np.logspace(np.log10(-ubmax), np.log10(-lbmin),
                    num=nsamples)[::-1]
            logging.info(f"""
            Ignoring lbmax and ubmin since both lbmin and ubmax are negative.
            Sampling from logarithmic grid between {lbmin} and {ubmax}.""")

        elif lbmin > 0 and ubmax > 0:
            alphaNP_samples = np.logspace(np.log10(lbmin), np.log10(ubmax),
                    num=nsamples)
            logging.info(f"""
            Ignoring lbmax and ubmin since both lbmin and ubmax are positive.
            Sampling from logarithmic grid between {lbmin} and {ubmax}.""")

        elif lbmin < 0 and ubmax > 0:
            nnegsamples = nsamples // 2
            nposamples = nsamples - nnegsamples

            negrid = -np.logspace(
                np.log10(-lbmax) + logridfrac, np.log10(-lbmin) - logridfrac,
                num=nnegsamples)[::-1]
            posgrid = np.logspace(
                np.log10(ubmin) - logridfrac, np.log10(ubmax) + logridfrac,
                num=nposamples)

            logging.info(f"""
            lbmin={lbmin} is negative and ubmax={ubmax} is positive.
            Sampling from logarithmic grids between {min(negrid)} and {max(negrid)},
            and between {min(posgrid)} and {max(posgrid)}.""")

            alphaNP_samples = np.concatenate([negrid, posgrid])
        else:
            raise ValueError(f"Invalid bounds {lb, ub} in generate_alphaNP_samples")

    return np.sort(alphaNP_samples)


# loglikelihoods & when to compute them
##############################################################################

# 2 different mehtods for the decomposition of the covariance matrix

def choLL(absd, covmat, lam=0):
    """
    For a given sample of absd, with the covariance matrix covmat, compute the
    negative log-likelihood using the Cholesky decomposition of covmat.

    """
    chol_covmat, lower = cho_factor(covmat + lam * np.eye(covmat.shape[0]), lower=True)

    absd_covinv_absd = absd.dot(cho_solve((chol_covmat, lower), absd))

    logdet = 2 * np.sum(np.log(np.diag(chol_covmat)))

    return 0.5 * (logdet + absd_covinv_absd)


def spectraLL(absd, covmat, lam=0):
    """
    For a given sample of absd, with the covariance matrix covmat, compute the
    negative log-likelihood using the spectral decomposition of covmat.
    """
    covmat += lam * np.eye(covmat.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covmat)

    inv_eigenvalues = 1.0 / eigenvalues
    covinv = eigenvectors @ np.diag(inv_eigenvalues) @ eigenvectors.T

    absd_covinv_absd = absd.dot(covinv).dot(absd)

    logdet = np.sum(np.log(eigenvalues))

    return 0.5 * (logdet + absd_covinv_absd)


# when to compute them

def get_llist_elemsamples(absdsamples, cov_decomp_method="cholesky", lam=0.):
    """
    Since the loglikelihood is estimated numerically, it requires a list of
    samples of the input parameters.

    Args:
        absdsamples:      list of absd samples (assumed to have been computed
                          for a fixed value of alphaNP and varying input
                          parameters)
    cov_decomp_method:    string specifying method with which the covariance
                          matrix is decomposed

    Returns:
        llist (np.array): np.array of negative loglikelihoods.

    """
    # estimate covariance matrix using absdsamples, computed for fixed alpha value

    cov_absd = np.cov(np.array(absdsamples), rowvar=False)

    if cov_decomp_method == "cholesky":
        LL = choLL
    elif cov_decomp_method == "spectral":
        LL = spectraLL

    llist = []

    for absd in absdsamples:
        llist.append(LL(absd, cov_absd, lam=lam))

    return np.array(llist)


def logL_alphaNP(alphaNP,
                 elem_collection,
                 nelemsamples,
                 min_percentile,
                 symm=False):
    """
    For elem_collection, compute negative loglikelihood for fixed alphaNP from
    ``nelemsamples`` samples of the input parameters associated to the elements
    of elem_collection.

    Args:
        alphaNP:          fixed alphaNP value
        elem_collection:  element collection of interest
        nelemsamples:     number of samples of the input parameters to be used
                          for estimation of loglikelihood.
        min_percentile:   percentile of samples to be used in comuptation of
                          minimum log-likelihood value min_ll
    Returns:
        min_ll:           log-likelihood value at min_percentile

    """

    for elem in elem_collection.elems:
        fitparams = elem.means_fit_params
        fitparams[-1] = alphaNP
        elem._update_fit_params(fitparams)

    loss = np.zeros(nelemsamples)

    for elem in elem_collection.elems:
        absdsamples = []

        inputparamsamples, fitparamsamples = generate_elemsamples(
            elem, nelemsamples)

        for i, inputparams in enumerate(inputparamsamples):
            # generate_elemsamples(elem, nelemsamples)
            elem._update_elem_params(inputparams)
            fitparams = fitparamsamples[i]
            fitparams.append(alphaNP)
            elem._update_fit_params(fitparams)
            absdsamples.append(elem.absd(symm))

        lls = get_llist_elemsamples(np.array(absdsamples),
            cov_decomp_method="cholesky")

        loss += lls

    return np.percentile(loss, min_percentile)  # np.mean(loss)


def compute_ll_experiments(
        elem_collection,
        alphasamples,
        nelemsamples,
        min_percentile,
        cov_decomp_method="cholesky"):

    """
    From ``nelemsamples`` samples of the input parameters associated to the
    elements of elem_collection, compute list of negative loglikelihoods for
    given experiment (list of alphaNP samples).

    Args:
        elem_collection:   collection of elements.
        alphasamples:      list of alphaNP values for which the loglikelihood is
                           to be computed.
        nelemsamples:      number of element samples
        cov_decomp_method: string specifying method with which the covariance
                           matrix entering the loglikelihood is decomposed.
    Returns:
        alphasamples (List[float]):   alphaNP samples
        lls (List[float]):            associated log likelihood values

    """

    lls = []

    # for each alpha in the list of generated alphas
    for alpha in alphasamples:
        # we collect a list of absd for each generated combination alpha-fitparams
        lls.append(logL_alphaNP(
            alphaNP=alpha,
            elem_collection=elem_collection,
            nelemsamples=nelemsamples,
            min_percentile=min_percentile)
        )

    return np.array(alphasamples), np.array(lls)


# compute delta chi^2 and its value corresponding to given number of sigmas

def get_delchisq(llist, minll=None):
    """
    Compute delta chi^2 from list of negative loglikelihoods, subtracting the
    minimum of the list.

    """
    if minll is None:
        minll = min(llist)

    if len(llist) > 0:
        llist = np.array(llist)
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


# determine bounds & their uncertainties
##############################################################################


# def minimise_logL_alphaNP(
#         elem_collection,
#         nelemsamples,
#         min_percentile,
#         maxiter,
#         opt_method,
#         bounds=(-1e-5, 1e-5),
#         tol=1e-12):
#     """
#     Scipy minimisation of negative loglikelihood as a function of alphaNP.
#
#     """
#
#     if opt_method == "annealing":
#         minlogL = dual_annealing(
#             logL_alphaNP,
#             bounds=[bounds],
#             args=(elem_collection, nelemsamples, min_percentile),
#             maxiter=maxiter)
#
#     elif opt_method == "differential_evolution":
#         minlogL = differential_evolution(
#             logL_alphaNP,
#             bounds=[bounds],
#             args=(elem_collection, nelemsamples, min_percentile),
#             maxiter=maxiter)
#
#     else:
#         minlogL = minimize(
#             logL_alphaNP,
#             x0=0,   # initial guess
#             bounds=[bounds],
#             args=(elem_collection, nelemsamples, min_percentile),
#             method=opt_method, options={"maxiter": maxiter},
#             tol=tol)
#
#     return minlogL
#

def blocking_bounds(
        messenger,
        lbs: List[float],
        ubs: List[float]):
    """
    Blocking method to compute the statistical uncertainty of the confidence
    intervals.

    Args:
        messenger:  specifies run configuration, in particular block_size (int,
                    number of points in each block)
        lbs (list): lower bounds determined by experiments.
        ubs (list): upper bounds determined by experiments.

    If the demanded block_size is larger than the number of surviving confidence
    intervals, the block_size is reduced to the number of surviving confidence
    intervals. In this case the uncertainties on the bounds cannot be computed.

    Returns:
        UB (float):       resulting upper bound
        LB (float):       resulting lower bound
        sig_UB (float):   uncertainty on UB
        sig_LB (float):   uncertainty on LB

    """

    if len(lbs) != len(ubs):
        raise ValueError("Lists of lower and upper bounds passed to the"
        "blocking method should be of equal length."
            f"len(lbs)={len(lbs)}, len(ubs)={len(ubs)}")

    block_size = messenger.params.block_size

    if block_size > len(lbs):
        block_size = len(lbs)
        logging.info(f"Reducing block size to {len(lbs)} since not enough samples "
            "have been generated. \n"
            "sig[LB] and sig[UB] cannot be computed in this case.")

    nblocks = int(len(lbs) / block_size)
    logging.info(f"nblocks {nblocks}")

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

    if messenger.params.verbose is True and block_size < len(lbs):

        from kifit.plot import blocking_plot

        blocking_plot(
            messenger,
            nblocks=nblocks,
            estimations=lb_val,
            uncertainties=sig_lb,
            label="Lower bound",
            plotname="blocking_lb"
        )
        blocking_plot(
            messenger,
            nblocks=nblocks,
            estimations=ub_val,
            uncertainties=sig_ub,
            label="Upper bound",
            plotname="blocking_ub"
        )
    return LB, UB, sig_LB, sig_UB


def get_bestalphaNP_and_bounds(
        messenger,
        bestalphaNPlist,
        confints):
    """
    Starting from the best alphaNP values determined by the experiments, apply
    the blocking method to compute the best alphaNP value, its uncertainty, as
    well as the nsigma - upper and lower bounds on alphaNP.

    Returns:
        best_alpha (float)
        sig_alpha (float)
        LB (float)
        sig_LB (float)
        UB (float)
        sig_UB (float)

    """
    confints = np.array(confints)

    best_alpha = np.median(bestalphaNPlist)
    sig_alpha = np.std(bestalphaNPlist)

    lowerbounds_exps = confints.T[0]
    lb_nans = np.argwhere(np.isnan(lowerbounds_exps))
    upperbounds_exps = confints.T[1]
    ub_nans = np.argwhere(np.isnan(upperbounds_exps))

    np.alltrue(lb_nans == ub_nans)

    lowerbounds_exps = lowerbounds_exps[~np.isnan(lowerbounds_exps)]
    upperbounds_exps = upperbounds_exps[~np.isnan(upperbounds_exps)]

    if len(lowerbounds_exps) == 0 or len(upperbounds_exps) == 0:
        return (best_alpha, sig_alpha, np.nan, np.nan, np.nan, np.nan)

    LB, UB, sig_LB, sig_UB = blocking_bounds(
        messenger,
        lowerbounds_exps,
        upperbounds_exps)

    logging.info(f"Final result: {best_alpha} with bounds [{LB}, {UB}].")

    return (best_alpha, sig_alpha, LB, sig_LB, UB, sig_UB)


def get_confint(alphas, delchisqs, delchisqcrit):
    """
    From lists of alphas and delta-chi-squared values and the critical
    delta-chi-squared value associated to the number of sigmas specified by the
    run parameters, compute the confidence interval for alphaNP.

    Returns:
       np.array of shape (2, )

    """
    alphas_inside = alphas[np.argwhere(delchisqs < delchisqcrit).flatten()]
    logging.info(f"Npts in fit confidence interval: {len(alphas_inside)}")

    # Best 1% of points was used to define minll.
    # Make sure there are more than 1% of points below delchisqcrit.

    if len(alphas_inside) >= 2:
        return np.array([min(alphas_inside), max(alphas_inside)])

    else:
        return np.array([np.nan, np.nan])


# searches
##############################################################################

# def perform_optigrid_search(
#         elem_collection,
#         messenger,
#         xind):
#
#     alphaNP_init = elem_collection.elems[0].alphaNP_init
#     sig_alphaNP_init = elem_collection.elems[0].sig_alphaNP_init
#
#     import optuna
#
#     num_searches = messenger.params.num_optigrid_searches
#
#     best_alphas = []
#
#     for search in range(num_searches):
#         optigridsearch = optuna.create_study(
#             sampler=optuna.samplers.TPESampler(), direction='minimize')
#
#         # optgridsearch.optimize(objective, n_trials=nalphasamples_exp)
#         optigridsearch.optimize(lambda trial:
#             objective(trial,
#                 elem_collection,
#                 messenger.params.num_elemsamples_per_alphasample_search,
#                 messenger.params.min_percentile),
#             n_trials=messenger.params.num_alphasamples_search)
#
#         best_alphas.append(optigridsearch.best_trial.params['alphaNP'])
#
#     best_alpha = np.median(best_alphas)
#     sig_alpha = max(best_alphas) - min(best_alphas)
#
#     best_delchisqs = []
#
#     LB = best_alpha - sig_alpha
#     sig_LB = None
#     UB = best_alpha + sig_alpha
#     sig_UB = None
#
#     for elem in elem_collection.elems:
#         elem.set_alphaNP_init(best_alpha, sig_alpha)
#
#     from kifit.plot import plot_search_output
#
#     plot_search_output(
#         messenger,
#         alphalist=best_alphas,
#         delchisqlist=np.zeros(len(best_alphas)),
#         delchisqcrit=None,
#         searchlims=[LB, UB],
#         xind=xind,
#         logplot=True
#     )
#
#     return [
#         alphaNP_init, sig_alphaNP_init,
#         np.array(best_alphas), np.array(best_delchisqs),
#         best_alpha, sig_alpha,
#         LB, sig_LB, UB, sig_UB,
#         None,
#         xind
#     ]


def get_lims_detlogrid(
        elem_collection,
        messenger,
        xind):

    det_maxpos = []
    det_minpos = []
    det_maxneg = []
    det_mineg = []

    for elem in elem_collection.elems:
        for dim in set(messenger.params.gkp_dims + [3]):
            det_output = messenger.paths.read_det_output("gkp", dim, xind)
            det_maxpos.append(max(det_output["allpos"]))
            det_minpos.append(min(det_output["allpos"]))
            det_maxneg.append(max(det_output["allneg"]))
            det_mineg.append(min(det_output["allneg"]))

        for dim in messenger.params.nmgkp_dims:
            det_output = messenger.paths.read_det_output("nmgkp", dim, xind)
            det_maxpos.append(max(det_output["allpos"]))
            det_minpos.append(min(det_output["allpos"]))
            det_maxneg.append(max(det_output["allneg"]))
            det_mineg.append(min(det_output["allneg"]))

        for dim in messenger.params.proj_dims:
            det_output = messenger.paths.read_det_output("proj", dim, xind)
            det_maxpos.append(max(det_output["allpos"]))
            det_minpos.append(min(det_output["allpos"]))
            det_maxneg.append(max(det_output["allneg"]))
            det_mineg.append(min(det_output["allneg"]))

    init_maxpos = np.nanmax(det_maxpos)
    init_minpos = np.nanmin(det_minpos)
    init_maxneg = np.nanmax(det_maxneg)
    init_mineg = np.nanmin(det_mineg)

    if np.isnan(init_maxpos) or np.isnan(init_mineg):
        init_maxpos = np.nanmax([init_maxpos, np.abs(init_mineg)])
        init_mineg = - init_maxpos

    if np.isnan(init_minpos) or np.isnan(init_maxneg):
        init_minpos = np.nanmax([init_minpos, np.abs(init_maxneg)])
        init_maxneg = - init_minpos

    return init_mineg, init_maxneg, init_minpos, init_maxpos


def determine_search_interval(
        elem_collection,
        messenger,
        xind):

    # if messenger.params.search_mode == "optigrid":
    #
    #     search_output = perform_optigrid_search(
    #         elem_collection=elem_collection,
    #         messenger=messenger,
    #         xind=xind)

    if messenger.params.search_mode == "detlogrid":

        (lbmin, lbmax, ubmin, ubmax) = get_lims_detlogrid(
            elem_collection=elem_collection,
            messenger=messenger,
            xind=xind)

        search_output = perform_experiments(
            elem_collection=elem_collection,
            messenger=messenger,
            xind=xind,
            expstr="search",
            lbmin=lbmin, lbmax=lbmax,
            ubmin=ubmin, ubmax=ubmax
        )

    else:

        logging.info("Global logrid search to determine search interval.")

        search_output = perform_experiments(
            elem_collection=elem_collection,
            messenger=messenger,
            xind=xind,
            expstr="search")

    return search_output


# experiments
##############################################################################

def organise_search_results(messenger, nexps, alphas, delchisqs, bestalphas, xind):

    # alphas = np.array(alphas).flatten()
    # delchisqs = np.array(delchisqs).flatten()

    best_alpha = np.median(bestalphas)

    # take larger of the two:
    #  - interval defined by median delchisq samples
    #  - 5 sigma interval
    delchisqcrit_search = max(np.median(delchisqs), get_delchisq_crit(5))

    search_interval = np.array(
        [
            get_confint(alphas[s], delchisqs[s], delchisqcrit_search)
            for s in range(nexps)
        ])

    from kifit.plot import plot_search_output

    if (messenger.params.search_mode == "detlogrid"
            or messenger.params.search_mode == "globalogrid"):
        logplot = True
    else:
        logplot = False

    plot_search_output(
        messenger,
        alphalist=alphas,
        delchisqlist=delchisqs,
        delchisqcrit=delchisqcrit_search,
        searchlims=search_interval,
        xind=xind,
        logplot=logplot
    )

    sig_alpha = np.max([
        np.abs(np.nanmax(search_interval) - best_alpha),
        np.abs(best_alpha - np.nanmin(search_interval))])

    LB = best_alpha - sig_alpha
    sig_LB = None
    UB = best_alpha + sig_alpha
    sig_UB = None
    logging.info(f"best_alpha search stage: {best_alpha}({sig_alpha})")

    return (best_alpha, sig_alpha, LB, sig_LB, UB, sig_UB)


def perform_experiments(
        elem_collection,
        messenger,
        xind=0,
        expstr="experiment",
        lbmin=None, lbmax=None,
        ubmin=None, ubmax=None):
    """
    Perform the experiments for the element collection elem_collection and the
    configuration specified by the messenger.

    Args:
        elem_collection:   element collection (instance of ElemCollection class)
        messenger:         run configuration (instance of Config class),
        xind (int):        x-index
        expstr (str):      if "experiment", standard experiment procedure is run,
                           if "search", the method is used as support method to
                           compute the initial search stage, which is determining
                           the interesting window to search for NP

    Returns:
        fit output (list): list of results that are also written to the fit
                           output file specified by the messenger.
                           N.B.: This output should fit to the fit_keys
                           specified in fitools.py
    """
    min_percentile = messenger.params.min_percentile

    if expstr == "search":

        nexps = 1
        nelemsamples_exp = messenger.params.num_elemsamples_per_alphasample_search
        nalphasamples_exp = messenger.params.num_alphasamples_search
        sample_mode = messenger.params.search_mode

        logging.info(f"{sample_mode} search to determine search interval.")

    elif expstr == "experiment":
        nexps = messenger.params.num_exp
        nelemsamples_exp = messenger.params.num_elemsamples_exp
        nalphasamples_exp = messenger.params.num_alphasamples_exp
        sample_mode = "normal"

    # we can use one of the elements as reference, since alphaNP is shared

    # also save output of search stage to data file
    alphaNP_init = elem_collection.elems[0].alphaNP_init
    sig_alphaNP_init = elem_collection.elems[0].sig_alphaNP_init

    allalphasamples = generate_alphaNP_samples(
        elem_collection.elems[0],
        nexps * nalphasamples_exp,
        search_mode=sample_mode,
        lbmin=lbmin, lbmax=lbmax,
        ubmin=ubmin, ubmax=ubmax,
        logridfrac=messenger.params.logrid_frac)

    # shuffle the sample
    np.random.shuffle(allalphasamples)

    # want all experiments (-> setups of the elemsamples) to be treated on equal
    # footing -> compute delchisq separately

    alphas_exps, lls_exps, bestalphas_exps, delchisqs_exps = [], [], [], []

    for exp in range(nexps):
        logging.info(f"Running {expstr} {exp+1}/{nexps}")

        # collect data for a single experiment
        alphasamples = allalphasamples[
            exp * nalphasamples_exp: (exp + 1) * nalphasamples_exp
        ]

        # compute alphas and LLs for this experiment
        alphas, lls = compute_ll_experiments(
            elem_collection,
            alphasamples,
            nelemsamples_exp,
            min_percentile)

        alphas_exps.append(alphas)
        bestalphas_exps.append(alphas[np.argmin(lls)])
        lls_exps.append(lls)

        if expstr == "experiment":
            minll_1 = np.percentile(lls, min_percentile)
            delchisqlist = get_delchisq(lls, minll=minll_1)

        elif expstr == "search":
            delchisqlist = get_delchisq(lls, minll=None)

        if messenger.params.verbose is True:

            from kifit.plot import plot_mc_output

            if (sample_mode == "detlogrid" or sample_mode == "globalogrid"):
                logplot = True
            else:
                logplot = False

            plot_mc_output(
                messenger,
                alphalist=alphas,
                delchisqlist=delchisqlist,
                expstr=expstr,
                plotname=f"{expstr}_{exp}",
                xind=xind,
                logplot=logplot
            )

        delchisqs_exps.append(delchisqlist)

    nsigmas = messenger.params.num_sigmas

    # using results of all experiments, compute confidence intervals

    if expstr == "experiment":
        delchisqcrit = get_delchisq_crit(nsigmas)

        confints_exps = np.array(
            [
                get_confint(alphas_exps[s], delchisqs_exps[s], delchisqcrit)
                for s in range(nexps)
            ])

        (best_alpha, sig_alpha, LB, sig_LB, UB, sig_UB) = \
            get_bestalphaNP_and_bounds(
                messenger,
                bestalphas_exps,
                confints_exps)

    # using results of search phase, determine search interval

    elif expstr == "search":

        (best_alpha, sig_alpha,
            LB, sig_LB, UB, sig_UB) = organise_search_results(
                messenger,
                nexps, alphas_exps, delchisqs_exps, bestalphas_exps, xind)

    else:
        raise ValueError(f"Invalid expstr {expstr}.")

    for elem in elem_collection.elems:
        elem.set_alphaNP_init(best_alpha, sig_alpha)

    return [
        alphaNP_init, sig_alphaNP_init,
        np.array(alphas_exps), np.array(delchisqs_exps),
        best_alpha, sig_alpha,
        LB, sig_LB, UB, sig_UB,
        nsigmas,
        xind
    ]


# full fitting procedure
##############################################################################

def sample_alphaNP_fit(
        elem_collection,
        messenger,
        xind: int = 0,
        verbose: bool = True):
    """
    Generate all fit data. This function specifies the fit procedure: In a first
    step, the search interval of interest is determined by performing a number
    of searches

    Args:
        elem_collection: element collection (instance of the ElemCollection class)
        messenger:       specifies the configuration (instance of the Config class)
        xind:            index of the X-coefficients for which the samples are
                         to be generated.

    Returns:
        fit output (list): list of results that are also written to the fit
                           output file specified by the messenger.
                           N.B.: This output should fit to the fit_keys
                           defined in fitools.py

    """
    logging.info(f"Performing King fit for x={xind}")

    for elem in elem_collection.elems:
        elem._update_Xcoeffs(xind)

    logging.info(f"scipy minimisation for x={xind}")

    search_output = determine_search_interval(
        elem_collection=elem_collection,
        messenger=messenger,
        xind=xind
    )

    messenger.paths.write_search_output(xind, search_output)

    logging.info(f"Experiments for x={xind}")

    fit_output = perform_experiments(
        elem_collection=elem_collection,
        messenger=messenger,
        xind=xind
    )

    messenger.paths.write_fit_output(xind, fit_output)

    return fit_output


# collect all data for mphi-vs-alphaNP plot
##############################################################################

def collect_fit_X_data(messenger):
    """
    Load all fit data produced in run specified by messenger (instance of the
    Config class) and organise it in terms of the X-coefficients.

    Returns:
        a set of lists, each of which has the length of the mphi-vector
        specified in the files of X-coefficients.

    """
    UB = []
    sig_UB = []
    LB = []
    sig_LB = []
    best_alphas = []
    sig_best_alphas = []

    for x in messenger.x_vals_fit:

        fit_output = messenger.paths.read_fit_output(x)

        UB.append(fit_output["UB"])
        sig_UB.append(fit_output["sig_UB"])
        LB.append(fit_output["LB"])
        sig_LB.append(fit_output["sig_LB"])
        best_alphas.append(fit_output["best_alpha"])

        sig_best_alphas.append(fit_output["sig_best_alpha"])

    return (np.array(UB), np.array(sig_UB),
            np.array(LB), np.array(sig_LB),
            np.array(best_alphas), np.array(sig_best_alphas))
