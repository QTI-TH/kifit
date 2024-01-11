import numpy as np

from scipy.odr import ODR, Model, RealData
from scipy.linalg import cholesky
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.stats import linregress, multivariate_normal, chi2

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
    sig_ph1s = np.array([sig_betas[j, 0] / (1 + betas[j, 0]) for j in
        range(len(betas))])

    kperp1s = betas.T[1] * np.cos(ph1s)
    sig_kperp1s = np.sqrt((sig_betas.T[1] * np.cos(ph1s))**2
            + (betas.T[1] * sig_ph1s * np.sin(ph1s))**2)

    return (betas, sig_betas,
            kperp1s, ph1s, sig_kperp1s, sig_ph1s)


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
    sig_kperp1s = np.sqrt((sig_betas.T[1] * np.cos(ph1s))**2
            + (betas.T[1] * sig_ph1s * np.sin(ph1s))**2)

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

    ll = (np.dot(A, A) / 2 + np.dot(At, At) / 2
        + np.sum(np.log(np.diag(chol_covmat)))
        + len(absd) / 2 * np.log(2 * np.pi))

    return ll


def get_llist(absdsamples, nsamples):
    """
    Get ll for list of absd-samples.

    """
    if (len(absdsamples) != nsamples):
        raise ValueError("""Mismatch between nsamples and length of absdsamples.""")
    cov_absd = np.cov(np.array(absdsamples), rowvar=False)

    llist = []
    for s in range(nsamples):
        llist.append(choLL(absdsamples[s], cov_absd))

    return np.array(llist)


def get_ll_for_varying_alphaNP(elem, nsamples):
    """
    Get a set of nsamples samples of absd by varying alphaNP only.

       alphaNP ~ N(0, sig[alphaNP_init])

    """
    fitparams = elem.means_fit_params
    sigalphaNP = elem.sig_alphaNP_init
    fitparamsamples = np.tensordot(np.ones(nsamples), fitparams, axes=0)
    alphaNPsamples = np.random.normal(fitparams[-1], sigalphaNP, nsamples)
    fitparamsamples[:, -1] = alphaNPsamples
    absdsamples = []
    for s in range(nsamples):
        print_progress(s, nsamples)
        elem._update_fit_params(fitparamsamples[s])
        absdsamples.append(elem.absd)

    return (get_llist(absdsamples, nsamples), alphaNPsamples)


def get_ll_for_varying_elemparams_and_alphaNP(elem, nsamples):
    """
    Get a set of nsamples samples of absd by varying the masses and isotope
    shifts according to the means and standard deviations given in the input
    files, as well as alphaNP.

       m  ~ N(<m>,  sig[m])
       m' ~ N(<m'>, sig[m'])
       v  ~ N(<v>,  sig[v])

       alphaNP ~ N(0, sig[alphaNP_init])

    """
    elemparamsamples = get_paramsamples(elem.means_input_params,
            elem.stdevs_input_params, nsamples)

    fitparams = elem.means_fit_params
    sigalphaNP = elem.sig_alphaNP_init
    fitparamsamples = np.tensordot(np.ones(nsamples), fitparams, axes=0)
    alphaNPsamples = np.random.normal(fitparams[-1], sigalphaNP, nsamples)
    fitparamsamples[:, -1] = alphaNPsamples

    absdsamples = []
    for s in range(nsamples):
        print_progress(s, nsamples)
        elem._update_elem_params(elemparamsamples[s])
        elem._update_fit_params(fitparamsamples[s])
        absdsamples.append(elem.absd)

    return (get_llist(absdsamples, nsamples), elemparamsamples, alphaNPsamples)


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
        raise ValueError("""delchisqlist provided to get_confints is not a
        delchisqlist. (Did you forget to subtract the minimum?)""")

    delchisq_crit = get_delchisq_crit(nsigmas, dof)

    pos = np.argwhere(delchisqlist < delchisq_crit).flatten()

    if len(pos) !=2:
        print("For " + str(nsigmas)
            + " sigma I found following limits for your confidence intervals:")
        print(np.sort(paramlist[pos]))

    return delchisq_crit, paramlist[pos]
