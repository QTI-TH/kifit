import os
from copy import deepcopy

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
    return a*x**2 + b*x + c

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


def sample_alphaNP_fit_fixed_elemparams(elem, nsamples, mphivar=False):
    """
    Keeping the element parameters fixed to their mean values, generate nsamples
    of

       alphaNP ~ N(0, sig[alphaNP_init]).

    If mphivar=True, this procedure is repeated for all X-coefficients provided
    for elem.

    Returns two (nmphi, nsamples)-dimensional numpy arrays: one with the values
    of alphaNP, the other with the respective loglikelihoods.

    """
    print(
        """Generating %s samples for the orthogonal distance King fit with
    fixed element parameters. mphi is %s varied."""
        % (nsamples, ("" if mphivar else "not"))
    )

    fitparams = elem.means_fit_params
    fitparamsamples = np.tensordot(np.ones(nsamples), fitparams, axes=0)

    if mphivar:
        Nx = len(elem.Xcoeff_data)
    else:
        Nx = 1

    alphalist = []
    llist = []

    for x in tqdm(range(Nx)):
        elem._update_Xcoeffs(x)
        sigalphaNP = elem.sig_alphaNP_init
        alphaNPsamples = np.random.normal(fitparams[-1], sigalphaNP, nsamples)
        alphalist.append(alphaNPsamples)
        fitparamsamples[:, -1] = alphaNPsamples

        absdsamples = []
        for s in range(nsamples):
            elem._update_fit_params(fitparamsamples[s])
            absdsamples.append(elem.absd)
        llist.append(get_llist(np.array(absdsamples), nsamples))

    alphalist = elem.dnorm * np.array(alphalist)
    llist = elem.dnorm * np.array(llist)

    return alphalist, llist


def generate_element_sample(elem, nsamples: int):
    """
    Generate ``nsamples`` of ``elem`` varying the input parameters according
    to the provided standard deviations.
    """
    parameters_samples = get_paramsamples(
        elem.means_input_params, elem.stdevs_input_params, nsamples
    )
    return parameters_samples


def generate_alphaNP_sample(
    elem, nsamples: int, search_mode: str = "random", delta=None
):
    """
    Generate ``nsamples`` of alphaNP according to ``elem`` initial conditions.
    """
    init_alphaNP = elem.alphaNP
    if search_mode == "random":
        alphaNP_samples = np.random.normal(
            init_alphaNP, elem.sig_alphaNP_init, nsamples
        )
    elif search_mode == "grid":
        alphaNP_samples = np.linspace(
            init_alphaNP - delta, init_alphaNP + delta, nsamples
        )
        print(f"New search interval: [{init_alphaNP - delta}, {init_alphaNP + delta}]")

    return alphaNP_samples


def compute_sample_ll(
    elem,
    element_samples,
    save_sample: bool = False,
    search_mode: str = "random",
    delta: float = 0.5,
    parabolic_fit: bool = False,
):
    """
    Generate alphaNP list for element ``elem`` according to ``parameters_samples``.

    Args:
        elem (Elem): target element.
        nsamples (int): number of samples.

    Return:
        List[float], List[float]: alphaNP samples and list of associated log likelihood.
    """

    nsamples = len(element_samples)
    fit_params_samples = np.tile(elem.means_fit_params, (nsamples, 1))
    fit_params_samples[:, -1] = generate_alphaNP_sample(
        elem=elem, nsamples=nsamples, search_mode=search_mode, delta=delta
    )

    absdsamples = []

    for s in range(nsamples):
        elem._update_elem_params(element_samples[s])
        elem._update_fit_params(fit_params_samples[s])
        absdsamples.append(elem.absd)

    if save_sample:
        np.save(arr=element_samples, file="parameters_samples")
        np.save(arr=fit_params_samples, file="fit_params_samples")

    generated_alphaNP_list = fit_params_samples[:, -1]
    generated_ll_list = get_llist(absdsamples, nsamples)

    np.save(arr=generated_alphaNP_list, file="generated_alphas")
    np.save(arr=generated_ll_list, file="generated_ll")
    
    if parabolic_fit:
        from kifit.plotfit import plot_parabolic_fit
        delchisq_list = get_delchisq(generated_ll_list)

        # normalizing alphas
        scale_factor = max(generated_alphaNP_list) - min(generated_alphaNP_list)
        normed_alpha_list = (generated_alphaNP_list - min(generated_alphaNP_list)) / scale_factor

        # parabolic fit with normed alphas
        popt, _ = curve_fit(
            parabola, 
            normed_alpha_list, 
            delchisq_list,
            p0 = [1., 0., 0.]
        )

        # generate a lot of new normed alphas in (0, 1)
        validation_x_data = np.linspace(0, 1, 1000000)
        # evaluate parabola
        validation_y_data = parabola(validation_x_data, *popt)

        # save index of the minimum of the parabola
        minimum_index = np.argmin(validation_y_data)

        # reconstruct the best alphaNP applying the inverse of the normalization
        best_alphaNP = validation_x_data[minimum_index] * scale_factor + min(generated_alphaNP_list)

        # perform parabolic fit
        plot_parabolic_fit(
            normed_alpha_list,
            delchisq_list, 
            parabola(normed_alpha_list, *popt),
            parabola_a=popt[0]
        )

    if parabolic_fit:
        return generated_alphaNP_list, generated_ll_list, best_alphaNP, popt[0]

    return generated_alphaNP_list, generated_ll_list 


def calculate_grid_delta(alphaNP_list, ll_list, delta_alpha_ratio: float = 0.5):
    """
    Take the minimum LL value in ``ll_list``, check the asymmetry of the sample
    and define the next delta as ``data_ratio`` interval of the shortest wing.
    """
    best_ll_index = np.argmin(ll_list)
    nsamples = len(alphaNP_list)

    print(best_ll_index, nsamples)

    if best_ll_index >= int(nsamples / 2):
        critical_alphas_interval = alphaNP_list[best_ll_index:]
    else:
        critical_alphas_interval = alphaNP_list[:best_ll_index]

    new_extreme = int(len(critical_alphas_interval) * delta_alpha_ratio)
    delta = np.abs(alphaNP_list[best_ll_index] - critical_alphas_interval[new_extreme])
    return delta


def iterative_mc_search(
    elem,
    n_sampled_elems: int = 1000,
    mphivar: bool = False,
    delta_alpha_ratio: float = 0.8,
    niter: int = 3,
    nx: int = 0,
    # big_n = 10000,
):
    """Perform iterative search."""

    # set the mass index
    elem._update_Xcoeffs(nx)

    elem_history = []

    # initializing iterative MC parameters
    delta = elem.sig_alphaNP_init
    new_parabola_param_a = 1e8
    parabola_param_a = 1e8

    while (new_parabola_param_a > 150):
        # generate element sample
        element_samples = generate_element_sample(elem, n_sampled_elems)

        # generate sample and fit with a parabola and updated delta
        alphas, ll, new_best_alpha, new_parabola_param_a = compute_sample_ll(
            elem,
            element_samples=element_samples,
            search_mode="grid",
            delta=delta,
            parabolic_fit=True,
        )      

        # if we still in search
        if new_parabola_param_a > 150:
            delta = calculate_grid_delta(alphas, ll, delta_alpha_ratio)
            parabola_param_a = new_parabola_param_a
            elem.alphaNP = new_best_alpha

            elem_history.append(deepcopy(elem))

            import matplotlib.pyplot as plt
            plt.figure()
            plt.scatter(alphas, ll)
            plt.savefig(f"test_{parabola_param_a}.png")

            print(f"alphaNP: {elem.alphaNP}, delta: {delta}")


        else:
            elem = elem_history[-1]
            print(f"Breaking loop. Keeping alphaNP with param {parabola_param_a} and alpha: {elem.alphaNP}")

    
    print(f"Check value parameter: {parabola_param_a}")
    print(f"alphaNP: {elem.alphaNP}, delta: {delta}")

    best_alphas = []

    for i in tqdm(range(niter)):
        element_samples = generate_element_sample(elem, n_sampled_elems)
        alphas, ll, best_alpha, _ = compute_sample_ll(
            elem,
            element_samples=element_samples,
            search_mode="grid",
            delta=delta,
            parabolic_fit=True,
        ) 

        exit()

        best_alphas.append(best_alpha)
    
    alpha_mean = np.mean(best_alphas)
    alpha_std = np.std(best_alphas)

    return alpha_mean, alpha_std


def get_delchisq(llist):
    return 2 * (llist - np.min(llist))


def get_delchisq_crit(nsigmas=2, dof=1):
    """
    Get chi^2 level associated to nsigmas

    """

    conf_level = chi2.cdf(nsigmas**2, 1)

    return chi2.ppf(conf_level, dof)


def get_confints(paramlist, delchisqlist, delchisqcrit):
    """
    Get nsigmas-confidence intervals.

    # TODO: fix these docstrings

    Returns:
    delchisq_crit: Delta chi^2 value associated to nsigmas.
    paramlist[pos]: parameter values with Delta chi^2 values in the vicinity of
    delchisq_crit

    """
    if np.min(delchisqlist) != 0:
        raise ValueError(
            """delchisqlist provided to get_confints is not a
        delchisqlist. (Did you forget to subtract the minimum?)"""
        )

    pos = np.argwhere(delchisqlist < delchisqcrit).flatten()

    return paramlist[pos]


def interpolate_mphi_alphaNP_fit(mphis, alphas, kind="next"):
    pass



def sample_alphaNP_det(
    elem, dim, nsamples, mphivar=False, gkp=True, outputdataname="alphaNP_det"
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

    file_path = (
        _output_data_path
        + "/"
        + outputdataname
        + "_"
        + elem.id
        + "_"
        + str(dim)
        + "-dim_"
        + method_tag
        + "_"
        + str(nsamples)
        + "_samples.pdf"
    )

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
        sigalphaNPs = vals[:, int((vals.shape[1]) / 2) :]

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
            Nx = len(elem.Xcoeff_data)
        else:
            Nx = 1

        # mphi_list = []
        alphaNPs = []  # alphaNP list for best alphaNP and all
        sigalphaNPs = []

        for x in range(Nx):
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


def sample_alphaNP_fit(elem, nsamples, mphivar=False):
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

    """
    # elemparamsamples = get_paramsamples(elem.means_input_params,
    #         elem.stdevs_input_params, nsamples)
    elemparamsamples = generate_element_sample(elem, nsamples)
    fitparams = elem.means_fit_params
    fitparamsamples = np.tensordot(np.ones(nsamples), fitparams, axes=0)

    if mphivar:
        Nx = len(elem.Xcoeff_data)
    else:
        Nx = 1

    alphalist = []
    llist = []

    for x in tqdm(range(Nx)):
        elem._update_Xcoeffs(x)
        sigalphaNP = elem.sig_alphaNP_init
        alphaNPsamples = np.random.normal(fitparams[-1], sigalphaNP, nsamples)
        alphalist.append(alphaNPsamples)
        fitparamsamples[:, -1] = alphaNPsamples

        absdsamples = []

        for s in range(nsamples):
            elem._update_elem_params(elemparamsamples[s])
            elem._update_fit_params(fitparamsamples[s])
            absdsamples.append(elem.absd)
        llist.append(get_llist(np.array(absdsamples), nsamples))

    alphalist = elem.dnorm * np.array(alphalist)
    llist = elem.dnorm * np.array(llist)

    return alphalist, llist, np.array(elemparamsamples)
