import logging
from typing import List

import numpy as np

from scipy.linalg import cho_factor, cho_solve
from scipy.odr import ODR, Model, RealData
from scipy.stats import chi2, linregress, multivariate_normal, loguniform

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


# linear King fit
##############################################################################

def linfit(p, x):
    return p[0] * x + p[1]


def get_odr_residuals(p, x, y, sx, sy):

    v = 1 / np.sqrt(1 + p[0] ** 2) * np.array([-p[0], 1])
    z = np.array([x, y]).T
    sz = np.array([np.diag([sx[i] ** 2, sy[i] ** 2]) for i in range(len(x))])

    residuals = np.array([v @ z[i] - p[1] * v[1] for i in range(len(x))])
    sigresiduals = np.sqrt(np.array([v @ sz[i] @ v for i in range(len(x))]))

    return residuals, sigresiduals


def perform_linreg(isotopeshiftdata, reference_transition_index: int = 0):
    """
    Perform linear regression.

    Args:
        isotopeshiftdata (normalised. rows=isotope pairs, columns=trans.)
        reference_transition_index (int, default: first transition)

    Returns:
        betas:       fit parameters (p in linfit)
        sig_betas:   uncertainties on betas
        kperp1s:     Kperp_i1, i=2,...,m (assuming ref. transition index = 0)
        ph1s:        phi_i1, i=2,...,m (assuming ref. transition index = 0)
        sig_kperp1s: uncertainties on kperp1s
        sig_ph1s:    uncertainties on ph1s

    """

    x = isotopeshiftdata.T[reference_transition_index]
    y = np.delete(isotopeshiftdata, reference_transition_index, axis=1)

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


def perform_odr(isotopeshiftdata, sigisotopeshiftdata,
                reference_transition_index: int = 0):
    """
    Perform separate orthogonal distance regression for each transition pair.

    Args:
        isotopeshiftdata (normalised. rows=isotope pairs, columns=trans.)
        reference_transition_index (int, default: first transition)

    Returns:
        betas:       fit parameters (p in linfit)
        sig_betas:   uncertainties on betas
        kperp1s:     Kperp_i1, i=2,...,m (assuming ref. transition index = 0)
        ph1s:        phi_i1, i=2,...,m (assuming ref. transition index = 0)
        sig_kperp1s: uncertainties on kperp1s
        sig_ph1s:    uncertainties on ph1s
        cov_kperp1_ph1s: covariance matrices for (kperp1, ph1)
    """
    lin_model = Model(linfit)

    x = isotopeshiftdata.T[reference_transition_index]
    y = np.delete(isotopeshiftdata, reference_transition_index, axis=1)

    sigx = sigisotopeshiftdata.T[reference_transition_index]
    sigy = np.delete(sigisotopeshiftdata, reference_transition_index, axis=1)

    betas = []
    sig_betas = []
    cov_kperp1_ph1s = []

    for i in range(y.shape[1]):
        data = RealData(x, y.T[i], sx=sigx, sy=sigy.T[i])
        beta_init = np.polyfit(x, y.T[i], 1)
        odr = ODR(data, lin_model, beta0=beta_init)
        out = odr.run()

        # Extract beta and covariance matrix
        betas.append(out.beta)
        sig_betas.append(out.sd_beta)
        cov_beta = out.cov_beta

        # Calculate ph1 and kperp1
        ph1 = np.arctan(out.beta[0])
        kperp1 = out.beta[1] * np.cos(ph1)

        # Derivatives for the delta method
        d_kperp1_d_beta0 = -out.beta[1] * np.sin(ph1)
        d_kperp1_d_beta1 = np.cos(ph1)
        d_ph1_d_beta0 = 1 / (1 + out.beta[0]**2)
        d_ph1_d_beta1 = 0

        # Jacobian matrix J
        J = np.array([[d_kperp1_d_beta0, d_kperp1_d_beta1],
                      [d_ph1_d_beta0, d_ph1_d_beta1]])

        # Covariance matrix for (kperp1, ph1)
        cov_kperp1_ph1 = J @ cov_beta @ J.T
        cov_kperp1_ph1s.append(cov_kperp1_ph1)

    betas = np.array(betas)
    sig_betas = np.array(sig_betas)

    ph1s = np.arctan(betas.T[0])
    sig_ph1s = np.arctan(sig_betas.T[0])

    kperp1s = betas.T[1] * np.cos(ph1s)
    sig_kperp1s = np.sqrt(
        (sig_betas.T[1] * np.cos(ph1s)) ** 2
        + (betas.T[1] * sig_ph1s * np.sin(ph1s)) ** 2
    )

    return (betas, sig_betas, kperp1s, ph1s, sig_kperp1s, sig_ph1s, cov_kperp1_ph1s)


# generate samples
##############################################################################

def generate_paramsamples(means, stdevs, nsamples):
    """
    Generate ``nsamples`` by sampling the multivariate normal distribution
    specified by by means and stdevs.

    """
    return multivariate_normal.rvs(means, np.diag(stdevs**2), size=nsamples)


def generate_elemsamples(elem, nsamples: int, sample_fitparams:bool=False):
    """
    Generate ``nsamples`` of the input parameters associated to ``elem`` from
    the normal distributions defined by the means and standard deviations
    provided by the corresponding data files:


       m  ~ N(<m>,  sig[m])   for the nuclear masses
       v  ~ N(<v>,  sig[v])   for the transition frequencies


    """
    inputparams_samples = generate_paramsamples(
        elem.means_input_params, elem.stdevs_input_params, nsamples
    )

    # sampling fit params if needed
    if sample_fitparams:
        samples_per_transition = []
        for kp1, ph1, cov in zip(elem.kp1_init, elem.ph1_init, elem.cov_kperp1_ph1):
            samples_per_transition.append(np.random.multivariate_normal([kp1, ph1], cov, size=nsamples))
        fitparams_samples = []
        for i in range(nsamples):
            these_fitparams = []
            for j in range(len(elem.kp1_init)):
                these_fitparams.append(samples_per_transition[j][i][0])
            for j in range(len(elem.kp1_init)):
                these_fitparams.append(samples_per_transition[j][i][1])
            fitparams_samples.append(these_fitparams)
    else:
        fitparams_samples = None

    return inputparams_samples, fitparams_samples


def generate_alphaNP_samples(elem,
        nsamples: int,
        search_mode: str = "normal",
        lbmin: float = None, lbmax: float = None,
        ubmin: float = None, ubmax: float = None,
        logridfrac: float = -17):
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
    elif search_mode == "logrid":
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
            # print("lbmax                        ", lbmax)
            # print("np.log10(-lbmax)             ", np.log10(-lbmax))
            # print("np.log10(-lbmax) + logridfrac", np.log10(-lbmax) + logridfrac)
            # print("np.log10(-lbmin)             ", np.log10(-lbmin))
            # print("np.log10(ubmin)              ", np.log10(ubmin))
            # print("np.log10(ubmin) + logridfrac ", np.log10(ubmin) + logridfrac)
            # print("np.log10(ubmax)              ", np.log10(ubmax))

            negrid = -np.logspace(np.log10(-lbmax) + logridfrac, np.log10(-lbmin),
                num=nnegsamples)[::-1]
            posgrid = np.logspace(np.log10(ubmin) + logridfrac, np.log10(ubmax),
                num=nposamples)

            logging.info(f"""
            lbmin={lbmin} is negative and ubmax={ubmax} is positive.
            Sampling from logarithmic grids between {min(negrid)} and {max(negrid)},
            and between {min(posgrid)} and {max(posgrid)}.""")

            # print("negrid")
            # print(negrid)
            # print("posgrid")
            # print(posgrid)

            alphaNP_samples = np.concatenate([negrid, posgrid])
        else:
            raise ValueError(f"Invalid bounds {lb, ub} in generate_alphaNP_samples")
    #
    # print("min generated alphaNP samples", min(alphaNP_samples))
    # print("max generated alphaNP samples", max(alphaNP_samples))
    #
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

def get_llist_elemsamples(absdsamples, cov_decomp_method="cholesky"):
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
        llist.append(LL(absd, cov_absd))

    return np.array(llist)


def logL_alphaNP(alphaNP, elem_collection, nelemsamples, min_percentile, sample_fitparams=False):
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
        sample_fitparams (bool): if `True`, kperp1 and ph1 are sampled from a 
                           multinormal distribution computed from the initial 
                           parameter's defintion via ODR. If `False`, the fit 
                           parameters are kept fixed
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
        inputparams_samples, fitparams_samples = generate_elemsamples(elem, nelemsamples, sample_fitparams=True)

        for i, inputparam in enumerate(inputparams_samples):
            # generate_elemsamples(elem, nelemsamples)
            elem._update_elem_params(inputparam)
            if sample_fitparams:
                these_fitparams = fitparams_samples[i]
                these_fitparams.append(alphaNP)
                elem._update_fit_params(these_fitparams)
            absdsamples.append(elem.absd)

        lls = get_llist_elemsamples(np.array(absdsamples),
            cov_decomp_method="cholesky")

        loss += lls

    return np.percentile(loss, min_percentile)  # np.mean(loss)


def compute_ll_experiments(
        elem_collection,
        alphasamples,
        nelemsamples,
        min_percentile,
        cov_decomp_method="cholesky",
        sample_fitparams=False,
    ):

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
        sample_fitparams (bool): if `True`, kperp1 and ph1 are sampled from a 
                           multinormal distribution computed from the initial 
                           parameter's defintion via ODR. If `False`, the fit 
                           parameters are kept fixed

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
            min_percentile=min_percentile,
            sample_fitparams=sample_fitparams,)
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


def minimise_logL_alphaNP(
        elem_collection,
        nelemsamples,
        min_percentile,
        maxiter,
        opt_method,
        bounds=(-1e-5, 1e-5),
        tol=1e-12,
        sample_fitparams=False,
        ):
    """
    Scipy minimisation of negative loglikelihood as a function of alphaNP.

    """

    if opt_method == "annealing":
        minlogL = dual_annealing(
            logL_alphaNP,
            bounds=[bounds],
            args=(elem_collection, nelemsamples, min_percentile),
            maxiter=maxiter,
            sample_fitparams=sample_fitparams,
        )

    elif opt_method == "differential_evolution":
        minlogL = differential_evolution(
            logL_alphaNP,
            bounds=[bounds],
            args=(elem_collection, nelemsamples, min_percentile),
            maxiter=maxiter,
            sample_fitparams=sample_fitparams,
        )

    else:
        minlogL = minimize(
            logL_alphaNP,
            x0=0,   # initial guess
            bounds=[bounds],
            args=(elem_collection, nelemsamples, min_percentile),
            method=opt_method, options={"maxiter": maxiter},
            tol=tol,
            sample_fitparams=sample_fitparams,
        )

    return minlogL


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

def determine_search_interval(
        elem_collection,
        messenger,
        xind):
    # sampling `nsamples` new elements for each element in the collections

    # if messenger.params.init_globalopt:
    #     logging.info("Preliminary global optimization to find reasonable bounds")

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

    search_output = perform_experiments(
        elem_collection=elem_collection,
        messenger=messenger,
        xind=xind,
        expstr="search",
        search_mode="logrid",
        lbmin=init_mineg, lbmax=init_maxneg,
        ubmin=init_minpos, ubmax=init_maxpos
    )

    # best_alpha_list = []
    #
    # for search in range(nsearches):
    #
    #     logging.info(f"Iterative search {search + 1}/{nsearches}")
    #
    #     # we can use one of the elements as reference, since alphaNP is shared
    #     alphasamples = generate_alphaNP_samples(
    #         elem_collection.elems[0],
    #         nsearches * nalphasamples_search,
    #         search_mode="logrid"
    #     )

        # res_min = minimise_logL_alphaNP(
        #     elem_collection=elem_collection,
        #     nelemsamples=nelemsamples_search,
        #     min_percentile=messenger.params.min_percentile,
        #     maxiter=messenger.params.maxiter,
        #     opt_method=messenger.params.optimization_method,
        #     bounds=(init_LB, init_UB)
        # )
        #
        # if res_min.success:
        #     best_alpha_list.append(res_min.x[0])
        #
        # logging.info(f"res_min: {res_min}")

    # best_alpha = np.median(best_alpha_list)
    #
    # sig_best_alpha = (max(best_alpha_list) - min(best_alpha_list))
    #
    # for elem in elem_collection.elems:
    #     elem.set_alphaNP_init(best_alpha, sig_best_alpha)
    #
    # logging.info(f"best_alpha search stage: {best_alpha}({sig_best_alpha})")

    return search_output


# experiments
##############################################################################

def perform_experiments(
        elem_collection,
        messenger,
        xind=0,
        expstr="experiment",
        search_mode="normal",
        lbmin=None, lbmax=None,
        ubmin=None, ubmax=None,):
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
        search_mode (str): determine the search mode. It can be chosen among 
                           ["normal", "lingrid", "logrid"]

    Returns:
        fit output (list): list of results that are also written to the fit
                           output file specified by the messenger.
                           N.B.: This output should fit to the fit_keys
                           specified in fitools.py
    """
    min_percentile = messenger.params.min_percentile

    if expstr == "search":
        nexps = 1  # messenger.params.num_searches
        nelemsamples_exp = messenger.params.num_elemsamples_per_alphasample_search
        nalphasamples_exp = messenger.params.num_alphasamples_search
        search_mode = "logrid"
        logridfrac = messenger.params.logrid_frac

    elif expstr == "experiment":
        nexps = messenger.params.num_exp
        nelemsamples_exp = messenger.params.num_elemsamples_exp
        nalphasamples_exp = messenger.params.num_alphasamples_exp
        search_mode = "normal"
        logridfrac = None

    else:
        raise ValueError(f"Invalid expstr {expstr}.")

    # we can use one of the elements as reference, since alphaNP is shared

    # also save output of search stage to data file
    alphaNP_init = elem_collection.elems[0].alphaNP_init
    sig_alphaNP_init = elem_collection.elems[0].sig_alphaNP_init

    allalphasamples = generate_alphaNP_samples(
        elem_collection.elems[0],
        nexps * nalphasamples_exp,
        search_mode=search_mode,
        lbmin=lbmin, lbmax=lbmax,
        ubmin=ubmin, ubmax=ubmax,
        logridfrac=logridfrac)

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
            min_percentile, 
            sample_fitparams=messenger.params.sample_fitparams,
        )

        alphas_exps.append(alphas)
        bestalphas_exps.append(alphas[np.argmin(lls)])
        lls_exps.append(lls)

        if expstr == "experiment":
            minll_1 = np.percentile(lls, min_percentile)
            delchisqlist = get_delchisq(lls, minll=minll_1)

        elif expstr == "search":
            delchisqlist = get_delchisq(lls, minll=min(lls))

        if messenger.params.verbose is True or expstr == "search":

            from kifit.plot import plot_mc_output

            if search_mode == "logrid":
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

        logging.info(f"LB     x={xind}: {LB}")
        logging.info(f"sig_LB x={xind}: {sig_LB}")
        logging.info(f"UB     x={xind}: {UB}")
        logging.info(f"sig_UB x={xind}: {sig_UB}")

    elif expstr == "search":
        best_alpha = np.median(bestalphas_exps)
        delchisqs_search = delchisqs_exps + np.min(delchisqs_exps)
        median_delchisq = np.median(delchisqs_search)

        confints_search = np.array(
            [
                get_confint(alphas_exps[s], delchisqs_exps[s], median_delchisq)
                for s in range(nexps)
            ])

        sig_alpha = np.max([
            np.abs(np.nanmax(confints_search) - best_alpha),
            np.abs(best_alpha - np.nanmin(confints_search))])

        LB = best_alpha - sig_alpha
        sig_LB = None
        UB = best_alpha + sig_alpha
        sig_UB = None
        logging.info(f"best_alpha search stage: {best_alpha}({sig_alpha})")

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
