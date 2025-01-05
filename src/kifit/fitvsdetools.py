import numpy as np
from math import factorial

from kifit.build import Elem, LeviCivita
from kifit.fitools import generate_paramsamples

np.set_printoptions(precision=54)

def estimate_covnutil(elem, nsamples):

    m_samples = generate_paramsamples(elem.m_a_in, elem.sig_m_a_in, nsamples)
    mp_samples = generate_paramsamples(elem.m_ap_in, elem.sig_m_ap_in, nsamples)
    mu_samples = 1 / m_samples - 1 / mp_samples

    nu_samples = generate_paramsamples(
            (elem.nu_in).flatten(), (elem.sig_nu_in).flatten(),
            nsamples).reshape(nsamples, elem.nisotopepairs, elem.ntransitions)

    nutil_samples = nu_samples / mu_samples[:, :, np.newaxis]

    vectorised_nutil = nutil_samples.reshape(nsamples,
                                             elem.nisotopepairs * elem.ntransitions)

    covectorised = np.cov(vectorised_nutil, rowvar=False)

    return covectorised
     #  .reshape(elem.nisotopepairs, elem.nisotopepairs, elem.ntransitions, elem.ntransitions)


def grad(func, x0, otherargs=[], h=1e-5):
    """
    Computes the gradient of a scalar function at a given point using finite differences.

    Parameters:
        func (callable): The scalar function to differentiate.
        x0 (array-like): The point at which to compute the gradient.
        h (float): Step size for finite differences.

    Returns:
        numpy.ndarray: The gradient vector.
    """
    x0 = np.asarray(x0)
    grad = np.zeros_like(x0)

    for i in range(len(x0)):

        x_forward = x0.copy()
        x_backward = x0.copy()

        x_forward[i] += h
        x_backward[i] -= h

        # print("x diff", x_forward[i] - x_backward[i])

        grad[i] = (func(x_forward, *otherargs) - func(x_backward, *otherargs)) / (2 * h)

    print()
    print("h", h)
    print("x0    ", x0)
    print("x_forward ", x_forward)
    print("x_backward", x_backward)
    print("x diff    ", x_forward - x_backward)
    print("f diff", func(x_forward, *otherargs) - func(x_backward, *otherargs))
    print()

    return grad

def alphaNP_GKP3_nutil(nutil, Xcoeffs, gammatil):

    dim = 3
    mutil = np.ones(nutil.shape[0])

    Vd = np.linalg.det(np.c_[nutil, mutil])

    V1 = 0
    for i, eps_i in LeviCivita(dim - 1):
        V1 += (eps_i * np.linalg.det(np.c_[
            Xcoeffs[i[0]] * gammatil,
            np.array([nutil[:, i[s]] for s in range(1, dim - 1)]).T,
            mutil]))

    return factorial(dim - 2) * np.array(Vd / V1)

def alphaNP_NMGKP3_nutil(nutil, Xcoeffs, gammatil):

    dim = 3
    mutil = np.ones(nutil.shape[0])

    Vd = np.linalg.det(nutil)

    V1 = 0
    for i, eps_i in LeviCivita(dim):
        V1 += (eps_i * np.linalg.det(np.c_[
            Xcoeffs[i[0]] * gammatil,
            np.array([nutil[:, i[s]] for s in range(1, dim)]).T]))

    return factorial(dim - 1) * np.array(vol_data / vol_alphaNP1)


def grad_alphaNP3(elem, ainds=[0, 1, 2], iinds=[0, 1], h=1e-5):

    dim = 3

    elem._update_elem_params(elem.means_input_params)
    mean_nutil = elem.nutil
    mean_alphaNP_GKP_combs = elem.alphaNP_GKP_combinations(dim)

    Xcoeffs = elem.Xvec[np.ix_(iinds)]
    gammatil = elem.gammatilvec[np.ix_(ainds)]

    assert np.isclose(
            elem.alphaNP_GKP(),
            alphaNP_GKP3_nutil(mean_nutil, Xcoeffs, gammatil),
            atol=0, rtol=1e-25)

    return grad(alphaNP_GKP3_nutil, mean_nutil, [Xcoeffs, gammatil], h=h).flatten()

def eFvec(elem, m=2):

    if m == elem.ntransitions:
        return elem.eF

    elif m < elem.ntransitions:
        F1 = elem.F1[:m]
        return F1 / np.sqrt(F1 @ F1)


def Pparallel(elem, n=3, m=2):

    eF = eFvec(elem)

    Pparallel = np.kron(np.eye(n), np.outer(eF, eF))

    assert np.allclose(Pparallel @ Pparallel, Pparallel, atol=0, rtol=1e-25)

    return Pparallel


def Pperp(elem, n=3, m=2):

    eF = eFvec(elem)

    Pperp = np.kron(np.eye(n), np.eye(m) - np.outer(eF, eF))

    print("Pperp @ Pperp", Pperp @ Pperp - Pperp)

    assert np.allclose(Pperp @ Pperp, Pperp, atol=0, rtol=1e-3)

    return Pperp


def SigmalphaNP(elem, nsamples):

    Sigma = estimate_covnutil(elem, nsamples)
    alpha_grad = grad_alphaNP3(elem, ainds=[0, 1, 2], iinds=[0, 1])

    P_perp = Pperp(elem)
    P_para = Pparallel(elem)

    print("alphagrad", alpha_grad.shape)
    print("Pperp    ", P_perp.shape)
    print("Sigma    ", Sigma.shape)

    SigmalphaNP = alpha_grad @ Sigma @ alpha_grad.T
    SigmalphaNP_perp = alpha_grad @ P_perp @ Sigma @ P_perp @ alpha_grad.T
    SigmalphaNP_parallel = alpha_grad @ P_para @ Sigma @ P_para @ alpha_grad.T

    print("Sigma test", SigmalphaNP - SigmalphaNP_perp - SigmalphaNP_parallel)

    return SigmalphaNP, SigmalphaNP_perp, SigmalphaNP_parallel

