import numpy as np

import scipy.stats as sps
from scipy.odr import ODR, Model, RealData

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

    slopes = []
    sig_slopes = []
    intercepts = []
    sig_intercepts = []

    for i in range(y.shape[1]):
        results = sps.linregress(x, y.T[i])
        slopes.append(results.slope)
        sig_slopes.append(results.stderr)
        intercepts.append(results.intercept)
        sig_intercepts.append(results.intercept_stderr)

    print("linreg slopes    ", slopes)
    print("linreg sig slopes", sig_slopes)
    print("linreg intercepts", intercepts)
    print("linreg sig interc", sig_intercepts)

    # ph1s = np.arctan(slopes)
    # kperp1s = intercepts * np.cos(ph1s)
    #

    # sig_ph1s = np.array([sig_slopes[j] / (1 + slope[j])])
    # # kperp1s = (betas.T[1] + mean_y) * np.cos(ph1s)
    # kperp1s = betas.T[1] * np.cos(ph1s)
    # # sig_kperp1s = np.sqrt((sig_betas.T[1] * np.cos(ph1s))**2
    # #         + ((betas.T[1] + mean_y) * sig_ph1s * np.sin(ph1s))**2)
    # sig_kperp1s = np.sqrt((sig_betas.T[1] * np.cos(ph1s))**2
    #         + (betas.T[1] * sig_ph1s * np.sin(ph1s))**2)

    # return (betas, sig_betas, kperp1s, ph1s, sig_kperp1s, sig_ph1s)

    return (np.array([slopes, intercepts]).T,
            np.array([sig_slopes, sig_intercepts]).T)


def perform_odr(isotopeshiftdata, sigisotopeshiftdata,
        reftrans_index: int = 0):
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
        print("i", i)
        print("sx", sigx)
        print("sy", sigy.T[i])
        data = RealData(x, y.T[i], sx=sigx, sy=sigy.T[i])
        results = sps.linregress(x, y.T[i])
        beta_init = [results.slope, results.intercept]
        print("beta_init", beta_init)

        odr = ODR(data, lin_model, beta0=beta_init)
        out = odr.run()
        betas.append(out.beta)
        sig_betas.append(out.sd_beta)

    betas = np.array(betas)
    sig_betas = np.array(sig_betas)

    print("beta_final ", np.array([betas.T[0], betas.T[1]]).T)

    ph1s = np.arctan(betas.T[0])
    sig_ph1s = np.arctan(sig_betas.T[0])

    kperp1s = betas.T[1] * np.cos(ph1s)
    sig_kperp1s = np.sqrt((sig_betas.T[1] * np.cos(ph1s))**2
            + (betas.T[1] * sig_ph1s * np.sin(ph1s))**2)

    return (betas, sig_betas, kperp1s, ph1s, sig_kperp1s, sig_ph1s)


# def perform_nodr(self, isotopeshiftdata, sigisotopeshiftdata,
#         reference_transition_index: int = 0):
#     """
#     Perform orthogonal distance regression for all transitions pairs.
#
#     Args:
#         data (normalised isotope shifts: rows=isotope pairs, columns=trans.)
#         reference_transition_index (default: first transition)
#
#     Returns:
#         slopes, intercepts, kperp1, ph1, sig_kperp1, sig_ph1
#
#     """
#     lin_model = Model(nlinfit)
#     beta_init = np.append(np.ones(isotopeshiftdata.shape[1] - 1), 0.)
#
#     print("beta_init", beta_init)
#
#     x = isotopeshiftdata.T[reference_transition_index]
#     y = np.delete(isotopeshiftdata, reference_transition_index, axis=1)
#
#     sigx = sigisotopeshiftdata.T[reference_transition_index]
#     sigy = np.delete(sigisotopeshiftdata, reference_transition_index, axis=1)
#
#
#     for i in range(y.shape[1]):
#         data = RealData(x, y[i], sx=sigx, sy=sigy[i])
#         odr = ODR(data, lin_model, beta0=beta_init)
#         out = odr.run()
#         betas.append(out.beta)
#         sig_betas.append(out.sd_beta)
#
#     betas = np.array(betas)
#     sig_betas = np.array(sig_betas)
#
#     ph1 = np.arctan(betas.T[1])
#     sig_ph1 = np.arctan(sig_betas.T[1])
#
#     kperp1 = betas.T[0] * np.cos(ph1)
#     sig_kperp1 = np.sqrt((sig_betas.T[0] * np.cos(ph1))**2
#             + (betas.T[0] * sig_ph1 * np.sin(ph1))**2)
#     return (betas, sig_betas, kperp1, ph1, sig_kperp1, sig_ph1)
#
#

