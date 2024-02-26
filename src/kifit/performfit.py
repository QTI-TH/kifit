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
    if (len(absdsamples[0]) != nsamples):
        raise ValueError("""Mismatch between nsamples and length of absdsamples.""")

    allist = []
    for x in range(len(absdsamples)):
        cov_absd = np.cov(np.array(absdsamples[x]), rowvar=False)

        llist = []
        for s in range(nsamples):
            llist.append(choLL(absdsamples[x, s], cov_absd))
        allist.append(llist)

    return np.array(allist)


def sample_ll_fixed_elemparams(elem, nsamples, mphivar=False):
    """
    Get a set of nsamples samples of absd by varying alphaNP only.

       alphaNP ~ N(0, sig[alphaNP_init])

    If mphivar=True, this procedure is repeated for all X-coefficients provided
    for elem.

    """
    print("""Generating %s samples for the orthogonal distance King fit with
    fixed element parameters. mphi is %s varied.""" % (nsamples,
        ("" if mphivar else "not")))

    fitparams = elem.means_fit_params
    sigalphaNP = elem.sig_alphaNP_init
    fitparamsamples = np.tensordot(np.ones(nsamples), fitparams, axes=0)
    alphaNPsamples = np.random.normal(fitparams[-1], sigalphaNP, nsamples)
    fitparamsamples[:, -1] = alphaNPsamples
    allabsdsamples = []

    if mphivar:
        Nx = len(elem.Xcoeff_data)
    else:
        Nx = 1

    allabsdsamples = []
    for x in range(Nx):
        if mphivar:
            elem._update_Xcoeffs(x)
            print_progress(x * nsamples, Nx * nsamples)
        absdsamples = []
        for s in range(nsamples):
            if not mphivar:
                print_progress(s, nsamples)
            elem._update_fit_params(fitparamsamples[s])
            absdsamples.append(elem.absd)
        allabsdsamples.append(absdsamples)

    return (get_llist(np.array(allabsdsamples), nsamples), alphaNPsamples)


def samplelem(elem, nsamples, orthkifit=True, genkifit=[], nmgenkifit=[],
        mphivar=False):
    """
    Get a set of nsamples samples of elem by varying the masses and isotope
    shifts according to the means and standard deviations given in the input
    files, as well as alphaNP.

       m  ~ N(<m>,  sig[m])
       m' ~ N(<m'>, sig[m'])
       v  ~ N(<v>,  sig[v])

       alphaNP ~ N(0, sig[alphaNP_init]).

    If mphivar=True, this procedure is repeated for all X-coefficients provided
    for elem.
    If orthkifit=True, the loglikelihood is computed using the orthogonal King
    fit.
    If genkifit=[dim1, dim2,...], alphaNP is computed using the (dim1,
    dim2,...)-dimensional Generalised King plot.
    If nmgenkifit=[dim1, dim2,...], alphaNP is computed using the (dim1,
    dim2,...)-dimensional no-mass Generalised King plot.

    """
    if genkifit==[]:
        gkf = False
    else:
        gkf = True

    if nmgenkifit==[]:
        nmgkf = False
    else:
        nmgkf = True

    elemparamsamples = get_paramsamples(elem.means_input_params,
            elem.stdevs_input_params, nsamples)

    if orthkifit:
        fitparams = elem.means_fit_params
        sigalphaNP = elem.sig_alphaNP_init
        print("testi", fitparams)

        fitparamsamples = np.tensordot(np.ones(nsamples), fitparams, axes=0)
        alphaNPsamples = np.random.normal(fitparams[-1], sigalphaNP, nsamples)
        fitparamsamples[:, -1] = alphaNPsamples

        allabsdsamples = []

    if mphivar:
        Nx = len(elem.Xcoeff_data)
    else:
        Nx = 1

    allalphaGKPsamples = []
    allalphaNMGKPsamples = []

    for x in range(Nx):
        if mphivar:
            elem._update_Xcoeffs(x)
            print_progress(x * nsamples, Nx * nsamples)
        if orthkifit:
            absdsamples = []
        if gkf:
            alphaGKPsamples = [[] for dim in genkifit]
        if nmgkf:
            alphaNMGKPsamples = [[] for dim in nmgenkifit]

        for s in range(nsamples):
            if not mphivar:
                print_progress(s, nsamples)

            elem._update_elem_params(elemparamsamples[s])

            if orthkifit:
                elem._update_fit_params(fitparamsamples[s])
                absdsamples.append(elem.absd)

            if gkf:
                for dim in genkifit:
                    # alphaGKPsamples[dim - 3].append(elem.alphaNP_GKP(dim))
                    alphaGKPsamples[dim - 3].append(elem.alphaNP_GKP(dim))

            if nmgkf:
                for dim in nmgenkifit:
                    # alphaNMGKPsamples[dim - 3].append(elem.alphaNP_NMGKP(dim))
                    alphaNMGKPsamples[dim - 3].append(elem.alphaNP_NMGKP(dim))

        if orthkifit:
            allabsdsamples.append(absdsamples)
        if gkf:
            allalphaGKPsamples.append(alphaGKPsamples)
        if nmgkf:
            allalphaNMGKPsamples.append(alphaNMGKPsamples)

    alphaNPllist = []
    if orthkifit:
        alphaNPllist.append([
            alphaNPsamples, get_llist(np.array(allabsdsamples), nsamples)])

    return (alphaNPllist, allalphaGKPsamples,
            allalphaNMGKPsamples, elemparamsamples)


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
    print("delchisq_crit", delchisq_crit)
    print("delchisqlist ", delchisqlist)

    print("""For %s sigma I found following limits for your confidence
            intervals:""" % nsigmas)
    print(np.sort(paramlist[pos]))

    return delchisq_crit, paramlist[pos]


def sample_alphaNP_GKP(elem, dim, nsamples, mphivar=False):
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
    print("""Using the %s-dimensional Generalised King Plot to compute
    alphaNP for %s samples. mphi is %svaried.""" % (dim, nsamples, ("" if
        mphivar else "not ")))
    print()
    print("""Initialising alphaNP GKP""")
    print()
    elemparamsamples = get_paramsamples(elem.means_input_params,
        elem.stdevs_input_params, nsamples)

    voldatsamples = []
    vol1samples = []

    # nutilsamples = []   # add mass-normalised isotope shifts for cross-check

    for s in range(nsamples):
        print_progress(s, nsamples)
        elem._update_elem_params(elemparamsamples[s])
        alphaNPparts = elem.alphaNP_GKP_part(dim)
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
    print("""Computing alphaNP GKP""")
    print()
    if mphivar:
        Nx = len(elem.Xcoeff_data)
    else:
        Nx = 1

    mphi_list = []
    alphaNP_list = []  # alphaNP list for best alphaNP and all
    sig_alphaNP_list = []

    for x in range(Nx):
        if mphivar:
            elem._update_Xcoeffs(x)
            mphi_list.append(elem.mphi)
            print_progress(nsamples * x, nsamples * Nx)

        """ p: alphaNP-permutation index and xpinds: X-indices for sample p"""
        nperm = len(xindlist)   # number of permutations of the i and a indices

        alphaNP_p_list = []
        sig_alphaNP_p_list = []
        for p, xpinds in enumerate(xindlist):
            if not mphivar:
                print_progress(p, nperm)
            meanvol1_p = np.array([elem.Xvec[xp] for xp in xpinds]) @ (meanvol1[p])
            sigvol1_p_sq = (np.array([elem.Xvec[xp]**2 for xp in xpinds])
                @ (sigvol1[p]**2))

            alphaNP_p_list.append(meanvoldat[p] / meanvol1_p)
            sig_alphaNP_p_list.append(
                (sigvoldat[p] / meanvol1_p)**2
                + (meanvoldat[p] / meanvol1_p**2)**2 * sigvol1_p_sq)
        alphaNP_list.append(alphaNP_p_list)
        sig_alphaNP_list.append(sig_alphaNP_p_list)

    alphaNP_list = np.math.factorial(dim - 2) * np.array(alphaNP_list)
    sig_alphaNP_list = np.math.factorial(dim - 2) * np.array(sig_alphaNP_list)

    return mphi_list, alphaNP_list, sig_alphaNP_list


#
#
# def sample_GKP_alphaNP(elem, dim, nsamples, mphivar=False):
#     """
#     Get a set of nsamples samples of alphaNP by varying the masses and isotope
#     shifts according to the means and standard deviations given in the input
#     files, as well as alphaNP.
#
#        m  ~ N(<m>,  sig[m])
#        m' ~ N(<m'>, sig[m'])
#        v  ~ N(<v>,  sig[v])
#
#     For each of these samples and for all possible combinations of the data,
#     compute alphaNP using the Generalised King Plot formula with
#
#         (nisotopepairs, ntransitions) = (dim, dim-1).
#
#     """
#     print("""Using the %s - dimensional Generalised King Plot to compute
#     alphaNP for %s samples. mphi is %s varied.""" % (dim, nsamples, ("" if
#         mphivar else "not")))
#
#     elemparamsamples = get_paramsamples(elem.means_input_params,
#         elem.stdevs_input_params, nsamples)
#
#     if mphivar:
#         Nx = len(elem.Xcoeff_data)
#     else:
#         Nx = 1
#
#     allalphaNPsamples = []
#     for x in range(Nx):
#         if mphivar:
#             elem._update_Xcoeffs(x)
#             print_progress(x * nsamples, Nx * nsamples)
#         alphaNPsamples = []
#         for s in range(nsamples):
#             if not mphivar:
#                 print_progress(s, nsamples)
#             elem._update_elem_params(elemparamsamples[s])
#             alphaNPsamples.append(elem.alphaNP_GKP(dim))
#         allalphaNPsamples.append(alphaNPsamples)
#
#     return np.array(allalphaNPsamples), elemparamsamples
#
#
# def sample_NMGKP_alphaNP(elem, dim, nsamples, mphivar=False):
#     """
#     Get a set of nsamples samples of alphaNP by varying the masses and isotope
#     shifts according to the means and standard deviations given in the input
#     files, as well as alphaNP.
#
#        m  ~ N(<m>,  sig[m])
#        m' ~ N(<m'>, sig[m'])
#        v  ~ N(<v>,  sig[v])
#
#     For each of these samples and for all possible combinations of the data,
#     compute alphaNP using the no-mass Generalised King Plot formula with
#
#         (nisotopepairs, ntransitions) = (dim, dim)
#
#     """
#     print("""Using the %s - dimensional no-mass Generalised King Plot to compute
#     alphaNP for %s samples. mphi is %s varied.""" % (dim, nsamples, ("" if
#         mphivar else "not")))
#     elemparamsamples = get_paramsamples(elem.means_input_params,
#         elem.stdevs_input_params, nsamples)
#
#     if mphivar:
#         Nx = len(elem.Xcoeff_data)
#     else:
#         Nx = 1
#
#     allalphaNPsamples = []
#     for x in range(Nx):
#         if mphivar:
#             elem._update_Xcoeffs(x)
#             print_progress(x * nsamples, Nx * nsamples)
#         alphaNPsamples = []
#         for s in range(nsamples):
#             if not mphivar:
#                 print_progress(s, nsamples)
#             elem._update_elem_params(elemparamsamples[s])
#             alphaNPsamples.append(elem.alphaNP_NMGKP(dim))
#         allalphaNPsamples.append(alphaNPsamples)
#
#     return np.array(allalphaNPsamples), elemparamsamples
