import numpy as np

from itertools import (
    product,
    combinations
)

from kifit.fitools import generate_paramsamples

# keys for saving / reading data
##############################################################################

det_keys = [
    "elem",
    "detstr",
    "dim",
    "alphas",
    "sigalphas",
    "npermutations",
    "minpos",
    "maxneg",
    "allpos",
    "allneg",
    "nsigmas",
    "x_ind"
]


# generate samples
##############################################################################

def sample_gkp_parts(
    elem,
    nsamples,
    dim,
    detstr='gkp'
):
    """
    Using the determinant formula specified by the dim and gkp arguments,
    generate mean & uncertainties of the numerator (voldat) and of the part of
    the denominator (vol1) that is independent of the X-coefficients.

    Args:
        elem:          element of interest (instance of the Elem class)
        nsamples (int):number of samples to be generated
        dim (int):     dimension of determinant
        gkp (boolean): if True, the formula associated with the generalised King
                       plot is used, else, that of the no-mass generalised King
                       plot is used.
        detstr (string): string specifying the type of determinant to be
                         computed. Valid options are "gkp", "nmgkp".

    Returns:
        meanvoldat:    mean values for numerator (voldat),
                       (np.array of dimension (number of data permutations, ))
        sigvoldat:     uncertainty on meanvoldat
                       (np.array of same dimension as meanvoldat)

        meanvol1:      mean values of contributions to denominator (vol1)
                       (np.array of dimension (number of data permutations,
                       number of non-zero entries in appropriate Levi-Civita
                       tensor))
        sigvol1:       uncertainties on meanvol1
                       (np.array of same dimension as meanvol1)
        xindlist:      list of indices of X-coefficients to be multiplied with
                       contributions to vol1
                       list of length = number of non-zero entries in
                       appropriate Levi-Civita tensor
    """

    if nsamples == 1:
        elemparamsamples = [elem.means_input_params]

    else:
        elemparamsamples = generate_paramsamples(
            elem.means_input_params, elem.stdevs_input_params, nsamples)

    voldatsamples = []
    vol1samples = []

    for s, paramsample in enumerate(elemparamsamples):
        elem._update_elem_params(paramsample)

        if detstr=="gkp":
            alphaNPparts = elem.alphaNP_GKP_part(dim)
        elif detstr=="nmgkp":
            alphaNPparts = elem.alphaNP_NMGKP_part(dim)

        else:
            raise ValueError("""Invalid detstr in sample_gkp_parts.
                Only gkp or nmgkp are valid here.""")

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

    return meanvoldat, sigvoldat, meanvol1, sigvol1, xindlist


def sample_proj_parts(
    elem,
    nsamples,
    dim
):
    """
    Using the determinant formula specified by the dim and gkp arguments,
    generate mean & uncertainties of the numerator (voldat) and of the part of
    the denominator (vol1) that is independent of the X-coefficients.

    Args:
        elem:          element of interest (instance of the Elem class)
        nsamples (int):number of samples to be generated
        dim (int):     dimension of determinant

    Returns:
        meanfrac:      mean values for X-independent part of the projection
                       (np.array of dimension (number of data permutations, ))
        sigfrac:       uncertainty on menafrac
                       (np.array of same dimension as meanvoldat)
        xindlist:      list of indices of Xij-coefficients. alphaNP is obtained
                       from alphaNP = frac / Xij.
                       (list of length (number of data permutations))
    """
    if nsamples == 1:
        elemparamsamples = [elem.means_input_params]

    else:
        elemparamsamples = generate_paramsamples(
            elem.means_input_params, elem.stdevs_input_params, nsamples)

    fracsamples = []

    for s, paramsample in enumerate(elemparamsamples):
        elem._update_elem_params(paramsample)

        fracombs, xinds = elem.alphaNP_proj_part(dim)

        fracsamples.append(fracombs)

        if s == 0:
            xindlist = xinds
        else:
            assert xindlist == xinds, (xindlist, xinds)

    # fracsamples has the form [sample][alphaNP-permutation]
    # for each term, average over all samples.

    meanfrac = np.average(np.array(fracsamples), axis=0)  # [permutation]
    sigfrac = np.std(np.array(fracsamples), axis=0)

    return meanfrac, sigfrac, xindlist


def assemble_gkp_combinations(
    elem,
    meanvoldat,
    sigvoldat,
    meanvol1,
    sigvol1,
    xindlist,
    dim,
    detstr="gkp"
):
    """
    Assemble alphaNP and sigma[alphaNP] from X-independent parts generated by
    sample_gkp_parts and appropriate X-coefficients.

    Args:
        elem:            Element of interest (instance of Elem class)
        meanvoldat:      mean numerator (computed by sample_gkp_parts)
        sigvoldat:       std numerator
        meanvol1:        means of X-independent parts of denominator
        sigvol1:         std X-independent parts of denominator
        xindlist (list): list of indices specifying which X-coefficients are to
                         be multiplied with the elements of vol1
        dim (int):       dimension of determinant
        detstr (string): string specifying the type of determinant to be
                         computed. Valid options are "gkp", "nmgkp".


    Returns:
        alphaNPs:        assembled alphaNP values
                         (np.array of dimension (number of data permutations, ))
        sigalphaNPs:     uncertainties on alphaNPs
                         (np.array of same dimensions as alphaNPs)

    """

    if detstr == "gkp":
        fac = np.math.factorial(dim - 2)
    elif detstr == "nmgkp":
        fac = np.math.factorial(dim - 1)
    else:
        raise ValueError("""Invalid detstr in assemble_gkp_combinations:
        Only gkp or nmgkp are valid here.""")

    alphalist = []
    sigalphalist = []

    """ p: alphaNP-permutation index and xpinds: X-indices for sample p"""
    for p, xpinds in enumerate(xindlist):
        vol1X = np.array([elem.Xvec[xp] for xp in xpinds]) @ (meanvol1[p])
        sigvol1X_sq = np.array([elem.Xvec[xp]**2 for xp in xpinds]) @ (
            sigvol1[p]**2)

        alphalist.append(meanvoldat[p] / vol1X)
        sigalphalist.append(np.sqrt(
            (sigvoldat[p] / vol1X) ** 2
            + (meanvoldat[p] / vol1X**2) ** 2 * sigvol1X_sq))

    alphaNPs = fac * np.array(alphalist)
    sigalphaNPs = fac * np.array(sigalphalist)

    return alphaNPs, sigalphaNPs


def assemble_proj_combinations(
    elem,
    meanfrac,
    sigfrac,
    xindlist
):
    """
    Assemble alphaNP and sigma[alphaNP] from X-independent parts generated by
    sample_proj_parts and the appropriate Xij-coefficients.

    Returns a list of p alphaNP-values, where p is the number of
    combinations of the data of dimension

       (nisotopepairs, 2) = (dim, 2),   dim >= 3.

    This part of the computation of alphaNP should be repeated for each mphi
    value.

    Args:
        elem:            Element of interest (instance of Elem class)
        frac:            mean X-independent contribution to alphaNP
                         (computed by sample_proj_parts)
        sigfrac:         std frac
        xindlist (list): list of indices specifying by which Xij-coefficients
                         the fracs should be divided (alphaNP = frac / Xij)
        dim (int):       "dimension" of projection formula
                         (corresponds to number of isotope pairs)

    Returns:
        alphaNPs:        assembled alphaNP values
                         (np.array of dimension (number of data permutations, ))
        sigalphaNPs:     uncertainties on alphaNPs
                         (np.array of same dimensions as alphaNPs)

    """

    alphalist = []
    sigalphalist = []

    """ p: alphaNP-permutation index and xpinds: X-indices for sample p"""
    for p, xind in enumerate(xindlist):
        alphalist.append(meanfrac[p] / elem.Xji(j=xind[1], i=xind[0]))
        sigalphalist.append(sigfrac[p] / elem.Xji(j=xind[1], i=xind[0]))

    return np.array(alphalist), np.array(sigalphalist)


def generate_alphaNP_dets(
    elem,
    messenger,
    dim,
    detstr="gkp"
):
    """
    Generate Get a set of ``nsamples`` of alphaNP by varying the masses and isotope
    shifts according to the means and standard deviations given in the input
    files:

       m  ~ N(<m>,  sig[m])
       m' ~ N(<m'>, sig[m'])
       v  ~ N(<v>,  sig[v])

    For each of these samples and for all possible combinations of the data,
    compute alphaNP using the Generalised King Plot formula with

        (nisotopepairs, ntransitions) = (dim, dim-1).

    Args:
        elem:          element of interest (instance of Elem class)
        messenger:     specifies run configuration (instance of Config class)
        dim (int):     dimension of (no-mass-) generalised King plot to be used
        gkp (boolean): if True, generalised King plot is computed, else no-mass
                       generalised King plot
    Returns:
        alphaNPs:      np.array of shape (lenp, ), with lenp (cf. below)
        sigalphas:     np.array of shape (lenp, ), with lenp (cf. below)
        lenp (int):    number of data permutations

    """
    nsamples = messenger.params.num_det_samples

    # GKP / NMGKP #############################################################

    if detstr == "gkp" or detstr == "nmgkp":
        # Part independent of X-coeffs
        meanvoldat, sigvoldat, meanvol1, sigvol1, xindlist = sample_gkp_parts(
            elem=elem,
            nsamples=nsamples,
            dim=dim,
            detstr=detstr)

        # Part with X-coeffs
        alphaNPs, sigalphaNPs = assemble_gkp_combinations(
            elem=elem,
            meanvoldat=meanvoldat,
            sigvoldat=sigvoldat,
            meanvol1=meanvol1,
            sigvol1=sigvol1,
            xindlist=xindlist,
            dim=dim,
            detstr=detstr)

        if detstr=="gkp":
            lenp = len(list(
                product(
                    combinations(elem.range_a, dim),
                    combinations(elem.range_i, dim - 1))))
        else:
            lenp = len(list(
                product(
                    combinations(elem.range_a, dim),
                    combinations(elem.range_i, dim))))

    # proj ####################################################################

    elif detstr=="proj":
        # Part independent of X-coeffs
        meanfrac, sigfrac, xindlist = sample_proj_parts(
            elem=elem,
            nsamples=nsamples,
            dim=dim)

        # Part with X-coeffs
        alphaNPs, sigalphaNPs = assemble_proj_combinations(
            elem=elem,
            meanfrac=meanfrac,
            sigfrac=sigfrac,
            xindlist=xindlist)

        lenp = len(list(
            product(
                combinations(elem.range_a, dim),
                combinations(elem.range_i, 2))))
    else:
        raise ValueError("""Invalid detstr in generate_alphaNP_dets.
            Only gkp, nmgkp or proj are valid.""")

    assert alphaNPs.shape[0] == lenp
    assert sigalphaNPs.shape[0] == lenp

    return alphaNPs, sigalphaNPs, lenp


# determine bounds
##############################################################################

def get_minpos_maxneg_alphaNP_bounds(alphaNPs, sigalphaNPs, nsigmas=2):
    """
    Determine smallest positive and largest negative values for the bound on
    alphaNP at the desired confidence level.

    all vectors have dimensions [x][perm]

    """
    alphaNPs = np.array(alphaNPs)  # [x][perm]
    sigalphaNPs = np.array(sigalphaNPs)  # [x][perm]

    assert all(sigalphaNPs > 0)

    alphaNP_UB = alphaNPs + nsigmas * sigalphaNPs
    alphaNP_LB = alphaNPs - nsigmas * sigalphaNPs

    allpos = np.where(alphaNP_UB > 0, alphaNP_UB, np.nan)
    allneg = np.where(alphaNP_LB < 0, alphaNP_LB, np.nan)

    minpos = np.nanmin(allpos)
    maxneg = np.nanmax(allneg)

    return minpos, maxneg, allpos, allneg


# det procedure
##############################################################################

def sample_alphaNP_det(
    elem,
    messenger,
    dim,
    detstr,
    xind=0
):
    """
    Generate determinant results for elem specified by

    Args:
        elem:          element of interest (instance of the Elem class)
        messenger:     run configuration (instance of the Config class)
        dim (int):     dimension of the (no-mass-) generalised King plot
        gkp (boolean): specifying whether the generalised King plot formula is
                       to be used (alternative: no-mass generalised King plot)
        xind (int):    index of the X-coefficient for which the results are to
                       be computed

    Returns:
        det_results_x: list of determinant reults which are also written to the
                       det output file specified by messenger.
                       N.B.: This output should fit to the det_keys defined in
                       detools.py


    """

    alphas, sigalphas, nb_permutations = generate_alphaNP_dets(
        elem=elem,
        messenger=messenger,
        dim=dim,
        detstr="gkp")

    (
        minpos, maxneg, allpos, allneg
    ) = get_minpos_maxneg_alphaNP_bounds(alphas, sigalphas,
        messenger.params.num_sigmas)

    det_results_x = [
        elem.id,
        detstr,
        dim,
        alphas,
        sigalphas,
        nb_permutations,
        minpos,
        maxneg,
        allpos,
        allneg,
        messenger.params.num_sigmas,
        xind
    ]

    messenger.paths.write_det_output(detstr, dim, xind, det_results_x)

    return det_results_x


# collect all data for mphi-vs-alphaNP plot
##############################################################################

def collect_det_X_data(config, dim, detstr):
    """
    Load all data produced in run specified by

    Args:
        messenger:     run configuration (instance of the Config class)
        dim (int):     dimension of the (no-mass-) generalised King plot
        gkp (boolean): indicates whether the generalised King plot formula is to
                       be used (alternative: no-mass generalised King plot

    ... and organise it in terms of the X-coefficients.

    Returns:
        a set of lists, each of which has the length of the mphi-vector
        specified in the files of X-coefficients.

        UB (np.array):     most stringent nsigma-upper bounds on alphaNP (with
                           nsigma specified in config)
        allpos (np.array): all positive nsigma-bounds on alphaNP
        LB (np.array):     most stringent nsigma-lower bounds on alphaNP
        allneg (np.array): all negative nsigma-bounds

    """
    alphas = []
    sigalphas = []
    UB = []
    allpos = []
    LB = []
    allneg = []

    for x in config.x_vals_det:

        det_output = config.paths.read_det_output(detstr, dim, x)

        lenp = det_output["npermutations"]

        alphas.append(det_output["alphas"])
        sigalphas.append(det_output["sigalphas"])
        UB.append(det_output["minpos"])
        allpos.append(det_output["allpos"])
        LB.append(det_output["maxneg"])
        allneg.append(det_output["allneg"])

    alphas = np.array(alphas).reshape(len(config.x_vals_det), lenp)
    sigalphas = np.array(sigalphas).reshape(len(config.x_vals_det), lenp)

    print("sig_alphaNP / alphaNP", np.mean(sigalphas / alphas, axis=1))

    allpos = np.array(allpos).reshape(len(config.x_vals_det), lenp)
    allneg = np.array(allneg).reshape(len(config.x_vals_det), lenp)

    UB = np.array(UB).reshape(len(config.x_vals_det))
    LB = np.array(LB).reshape(len(config.x_vals_det))

    return (alphas, sigalphas,
            UB, allpos, LB, allneg)

