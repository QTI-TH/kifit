import numpy as np
import os
from math import factorial

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


plotfolder = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "test_output"))

if not os.path.exists(plotfolder):
    os.mkdir(plotfolder)


# generate samples
##############################################################################

def sample_gkp_combinations(
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
        alphasamples:  np.array of alphaNP samples
                       (dimension (nsamples, number of data permutations))
        elemparamsamples: np.array of input parameter samples
                       (dimension (nsamples, ))
    """

    if nsamples == 1:
        elemparamsamples = np.array([elem.means_input_params])

    else:
        elemparamsamples = generate_paramsamples(
            elem.means_input_params, elem.stdevs_input_params, nsamples)

    # np.savetxt(os.path.join(plotfolder,
    #                         f"{elem.id}_elemparamsamples_Ns{nsamples}.txt"),
    #                elemparamsamples, delimiter=",")
    #

    alphasamples = []

    for s, paramsample in enumerate(elemparamsamples):
        elem._update_elem_params(paramsample)

        if detstr=="gkp":
            alphasamples.append(elem.alphaNP_GKP_combinations(dim))

        elif detstr=="nmgkp":
            alphasamples.append(elem.alphaNP_NMGKP_combinations(dim))

        else:
            raise ValueError("""Invalid detstr in sample_gkp_parts.
                Only gkp or nmgkp are valid here.""")

    alphasamples = np.array(alphasamples)

    return alphasamples, elemparamsamples


def sample_proj_combinations(
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
        alphasamples:  np.array of alphaNP samples
                       (dimension (number of data permutations, ))
    """
    if nsamples == 1:
        elemparamsamples = np.array([elem.means_input_params])

    else:
        elemparamsamples = generate_paramsamples(
            elem.means_input_params, elem.stdevs_input_params, nsamples)

    alphasamples = []

    for s, paramsample in enumerate(elemparamsamples):
        elem._update_elem_params(paramsample)
        alphasamples.append(elem.alphaNP_proj_combinations(dim))

    alphasamples = np.array(alphasamples)
    return alphasamples, elemparamsamples


def generate_alphaNP_dets(
    elem,
    nsamples,
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
        dim (int):     dimension of (no-mass-) generalised King plot to be used
        gkp (boolean): if True, generalised King plot is computed, else no-mass
                       generalised King plot
    Returns:
        alphaNPs:      np.array of shape (lenp, ), with lenp (cf. below)
        sigalphas:     np.array of shape (lenp, ), with lenp (cf. below)
        lenp (int):    number of data permutations

    """


    # GKP / NMGKP #############################################################

    if detstr == "gkp" or detstr == "nmgkp" and nsamples > 1:
        # Part independent of X-coeffs
        alphasamples, _ = sample_gkp_combinations(
            elem=elem,
            nsamples=nsamples,
            dim=dim,
            detstr=detstr)


    if detstr=="gkp":
        lenp = len(list(
            product(
                combinations(elem.range_a, dim),
                combinations(elem.range_i, dim - 1))))

        elem._update_elem_params(elem.means_input_params)
        meanalphas = elem.alphaNP_GKP_combinations(dim)  # [permutation]

    elif detstr=="nmgkp":
        lenp = len(list(
            product(
                combinations(elem.range_a, dim),
                combinations(elem.range_i, dim))))

        elem._update_elem_params(elem.means_input_params)
        meanalphas = elem.alphaNP_NMGKP_combinations(dim)  # [permutation]

    # proj ####################################################################

    elif detstr=="proj":
        # Part independent of X-coeffs

        if nsamples > 1:
            alphasamples, _ = sample_proj_combinations(
                elem=elem,
                nsamples=nsamples,
                dim=dim)

        lenp = len(list(
            product(
                combinations(elem.range_a, dim),
                combinations(elem.range_i, 2))))

        elem._update_elem_params(elem.means_input_params)
        meanalphas = elem.alphaNP_proj_combinations(dim)  # [permutation]

    else:
        raise ValueError("""Invalid detstr in generate_alphaNP_dets.
            Only gkp, nmgkp or proj are valid.""")

    # meanalphas = np.average(alphasamples, axis=0)  # [permutation]

    if nsamples > 1:
        sigalphas = np.std(alphasamples, axis=0)  # [permutation]
    else:
        sigalphas = np.empty((lenp))
        sigalphas[:] = np.nan

    assert meanalphas.shape[0] == lenp
    assert sigalphas.shape[0] == lenp

    return meanalphas, sigalphas, lenp


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
        nsamples=messenger.params.num_det_samples,
        dim=dim,
        detstr=detstr)

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

    allpos = np.array(allpos).reshape(len(config.x_vals_det), lenp)
    allneg = np.array(allneg).reshape(len(config.x_vals_det), lenp)

    UB = np.array(UB).reshape(len(config.x_vals_det))
    LB = np.array(LB).reshape(len(config.x_vals_det))

    return (alphas, sigalphas,
            UB, allpos, LB, allneg)

