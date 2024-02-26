import numpy as np
from scipy.linalg import cholesky
from scipy.odr import ODR, Model, RealData
from scipy.stats import chi2, linregress, multivariate_normal
from tqdm import tqdm

np.random.seed(1)


def linfit(p, x):
    return p[0] * x + p[1]


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


def choLL(absd, covmat):
    """
    For a given sample of absd, with the covariance matrix covmat, compute the
    log-likelihood using the Cholesky decomposition of covmat.

    """

    chol_covmat = cholesky(covmat, lower=True)

    A = np.linalg.solve(chol_covmat, absd)
    At = np.linalg.solve(chol_covmat.T, absd)

    ll = (
        np.dot(A, A) / 2
        + np.dot(At, At) / 2
        + np.sum(np.log(np.diag(chol_covmat)))
        + len(absd) / 2 * np.log(2 * np.pi)
    )

    return ll


def get_llist(absdsamples, nsamples):
    """
    Get ll for list of absd-samples.

    """
    if len(absdsamples[0]) != nsamples:
        raise ValueError("""Mismatch between nsamples and length of absdsamples.""")

    allist = []
    for x in range(len(absdsamples)):
        cov_absd = np.cov(np.array(absdsamples[x]), rowvar=False)

        llist = []
        for s in range(nsamples):
            llist.append(choLL(absdsamples[x, s], cov_absd))
        allist.append(llist)

    return np.array(allist)


def generate_element_sample(elem, nsamples: int):
    """
    Generate ``nsamples`` of ``elem`` varying the input parameters according
    to the provided standard deviations.
    """
    parameters_samples = get_paramsamples(
        elem.means_input_params, elem.stdevs_input_params, nsamples
    )
    return parameters_samples


def generate_alphaNP_sample(elem, nsamples: int):
    """
    Generate ``nsamples`` of alphaNP according to ``elem`` initial conditions.
    """
    init_alphaNP = elem.means_fit_params[-1]
    alphaNP_samples = np.random.normal(init_alphaNP, elem.sig_alphaNP_init, nsamples)
    return alphaNP_samples


def compute_sample_ll(elem, nsamples, mphivar: bool = False, save_sample: bool = False):
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

    # generate sample of elements
    parameters_samples = generate_element_sample(elem, nsamples)
    # fix fit parameters
    fit_params_samples = np.tile(elem.means_fit_params, (nsamples, 1))
    # varying just alphaNP
    fit_params_samples[:, -1] = generate_alphaNP_sample(elem=elem, nsamples=nsamples)

    allabsdsamples = []

    if mphivar:
        Nx = len(elem.Xcoeff_data)
    else:
        Nx = 1

    for x in tqdm(range(Nx)):
        if mphivar:
            elem._update_Xcoeffs(x)

        absdsamples = []

        for s in range(nsamples):
            elem._update_elem_params(parameters_samples[s])
            elem._update_fit_params(fit_params_samples[s])
            absdsamples.append(elem.absd)

    allabsdsamples.append(absdsamples)

    if save_sample:
        np.save(arr=parameters_samples, file="parameters_samples")
        np.save(arr=fit_params_samples, file="fit_params_samples")

    return fit_params_samples[:, -1], get_llist(np.array(allabsdsamples), nsamples)


def get_delchisq(llist):
    return 2 * (llist - np.min(llist))


def get_delchisq_crit(nsigmas=2, dof=1):
    """
    Get chi^2 level associated to nsigmas

    """

    conf_level = chi2.cdf(nsigmas**2, 1)

    return chi2.ppf(conf_level, dof)


def get_confints(paramlist, delchisqlist, nsigmas=2, dof=1):
    """
    Interpolate data in xvals & yvals and compute nsigmas-confidence intervals.

    dof: degrees of freedom used in computation of confidence intervals
    npts: number of points used for interpolation

    Returns:

    """
    if np.min(delchisqlist) != 0:
        raise ValueError(
            """delchisqlist provided to get_confints is not a
        delchisqlist. (Did you forget to subtract the minimum?)"""
        )

    delchisq_crit = get_delchisq_crit(nsigmas, dof)

    pos = np.argwhere(delchisqlist < delchisq_crit).flatten()
    print("delchisq_crit", delchisq_crit)
    print("delchisqlist ", delchisqlist)

    print(
        """For %s sigma I found following limits for your confidence
            intervals:"""
        % nsigmas
    )
    print(np.sort(paramlist[pos]))

    return delchisq_crit, paramlist[pos]


def sample_GKP_alphaNP(elem, dim, nsamples, mphivar=False):
    """
    Get a set of nsamples samples of alphaNP by varying the masses and isotope
    shifts according to the means and standard deviations given in the input
    files, as well as alphaNP.

       m  ~ N(<m>,  sig[m])
       m' ~ N(<m'>, sig[m'])
       v  ~ N(<v>,  sig[v])

    For each of these samples and for all possible combinations of the data,
    compute alphaNP using the Generalised King Plot formula with

        (nisotopepairs, ntransitions) = (dim, dim-1).

    """
    print(
        """Using the %s - dimensional Generalised King Plot to compute
    alphaNP for %s samples. mphi is %s varied."""
        % (dim, nsamples, ("" if mphivar else "not"))
    )

    elemparamsamples = get_paramsamples(
        elem.means_input_params, elem.stdevs_input_params, nsamples
    )

    if mphivar:
        Nx = len(elem.Xcoeff_data)
    else:
        Nx = 1

    allalphaNPsamples = []
    for x in range(Nx):
        if mphivar:
            elem._update_Xcoeffs(x)
            print_progress(x * nsamples, Nx * nsamples)
        alphaNPsamples = []
        for s in range(nsamples):
            if not mphivar:
                print_progress(s, nsamples)
            elem._update_elem_params(elemparamsamples[s])
            alphaNPsamples.append(elem.alphaNP_GKP(dim))
        allalphaNPsamples.append(alphaNPsamples)

    return np.array(allalphaNPsamples), elemparamsamples
