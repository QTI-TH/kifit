import numpy as np
from scipy.linalg import lu
from math import factorial
from itertools import combinations, product

from kifit.build import Elem, LeviCivita
from kifit.fitools import generate_paramsamples

np.set_printoptions(precision=54)


def prepare_dataset(elem, ainds, iinds):

    elem._update_elem_params(elem.means_input_params)

    nutil = elem.nutil[np.ix_(ainds, iinds)]
    mutil = np.ones(nutil.shape[0])
    Xcoeffs = elem.Xvec[np.ix_(iinds)]
    gammatil = elem.gammatilvec[np.ix_(ainds)]

    return nutil, Xcoeffs, gammatil, mutil


def covnutil_ai(elem, ainds, iinds, nsamples):

    m_a = elem.m_a_in[np.ix_(ainds)]
    sig_m_a = elem.sig_m_a_in[np.ix_(ainds)]
    m_ap = elem.m_ap_in[np.ix_(ainds)]
    sig_m_ap = elem.sig_m_ap_in[np.ix_(ainds)]

    nu = elem.nu_in[np.ix_(ainds, iinds)]
    sig_nu = elem.sig_nu_in[np.ix_(ainds, iinds)]

    nisotopepairs = len(ainds)
    ntransitions = len(iinds)


    m_samples = generate_paramsamples(m_a, sig_m_a, nsamples)
    mp_samples = generate_paramsamples(m_ap, sig_m_ap, nsamples)
    mu_samples = 1 / m_samples - 1 / mp_samples

    nu_samples = generate_paramsamples(
            nu.flatten(), sig_nu.flatten(),
            nsamples).reshape(nsamples, nisotopepairs, ntransitions)

    nutil_samples = nu_samples / mu_samples[:, :, np.newaxis]

    vectorised_nutil = nutil_samples.reshape(nsamples,
                                             nisotopepairs * ntransitions)

    covectorised = np.cov(vectorised_nutil, rowvar=False)


    return covectorised


def LU_det(mat):

    (sign, logabsdet) = np.linalg.slogdet(mat)
    if sign == 0:
        return -np.inf
    else:
        return sign * np.exp(logabsdet)


def normalise_columns(mat):
    norms = np.linalg.norm(mat, axis=0, keepdims=True)
    norms[norms == 0] = 1
    return mat / norms, norms.flatten()

def V1_GKP_nutil(nutil, Xcoeffs, gammatil):

    dim = nutil.shape[0]
    mutil = np.ones(dim)

    V1 = 0
    for i, eps_i in LeviCivita(dim - 1):
        V1 += (eps_i * LU_det(np.c_[
            Xcoeffs[i[0]] * gammatil,
            np.array([nutil[:, i[s]] for s in range(1, dim - 1)]).T,
            mutil]))
    return V1

def V1_GKP_nutil_normed(nutil, Xcoeffs, gammatil):

    dim = nutil.shape[0]
    mutil = np.ones(dim)

    nutil_normed, norms_nutil = normalise_columns(nutil)

    V1 = 0
    for i, eps_i in LeviCivita(dim - 1):
        det_mat = LU_det(np.c_[
            Xcoeffs[i[0]] * gammatil,
            np.array([nutil_normed[:, i[s]] for s in range(1, dim - 1)]).T,
            mutil])
        norm_factor = np.prod(norms_nutil[i[1:]])

        V1 += eps_i * det_mat * norm_factor

    return V1


def alphaNP_GKP_nutil(nutil, Xcoeffs, gammatil):

    dim = nutil.shape[0]
    mutil = np.ones(dim)

    Vd = LU_det(np.c_[nutil, mutil])

    V1 = V1_GKP_nutil(nutil, Xcoeffs, gammatil)

    return factorial(dim - 2) * np.array(Vd / V1)


def alphaNP_GKP_nutil_normed(nutil, Xcoeffs, gammatil):

    dim = nutil.shape[0]
    mutil = np.ones(dim)

    dim = nutil.shape[0]
    gradalpha = np.zeros_like(nutil)

    nutil_normed, norms_nutil = normalise_columns(nutil)
    nutilmat = np.c_[nutil_normed, mutil]

    Vd = LU_det(nutilmat) * np.prod(norms_nutil)
    V1 = V1_GKP_nutil_normed(nutil, Xcoeffs, gammatil)
    print("Vd normed", Vd)
    print("V1 normed", V1)


    return factorial(dim - 2) * np.array(Vd / V1)




def grad_alphaNP_nutil(nutil, Xcoeffs, gammatil, mutil, eps=1e-17):

    print("nutil", nutil)
    print("gammatil", gammatil)

    dim = nutil.shape[0]
    gradalpha = np.zeros_like(nutil)

    nutil_normed, norms_nutil = normalise_columns(nutil)
    nutilmat = np.c_[nutil_normed, mutil]

    Vd = LU_det(nutilmat) * np.prod(norms_nutil)
    V1 = V1_GKP_nutil(nutil, Xcoeffs, gammatil)

    print("Vd", Vd)
    print("V1", V1)

    for a in range(dim):
        for i in range(dim - 1):
            del_nutil = nutil.copy()
            del_nutil[:, i] = np.eye(dim)[a]

            dVd = LU_det(np.c_[del_nutil, mutil])
            dV1 = V1_GKP_nutil(del_nutil, Xcoeffs, gammatil)

            gradalpha[a, i] = Vd * dV1 / V1**2 * (V1 * dVd / (Vd * dV1) - 1)


            print("a", a)
            print("i", i)
            print("V1 dVd", V1 * dVd)
            print("Vd dV1", Vd * dV1)
            print("V1 dVd - Vd dV1", (V1 * dVd - Vd * dV1))
            print("1/V1^2", 1/ V1**2)
            print("gradalpha", (V1 * dVd - Vd * dV1) / V1**2)

            print("new frac", V1 * dVd / (Vd * dV1) )


    return gradalpha.flatten()


def V1_NMGKP_nutil(nutil, Xcoeffs, gammatil):

    dim = nutil.shape[0]

    V1 = 0
    for i, eps_i in LeviCivita(dim):
        V1 += (eps_i * LU_det(np.c_[
            Xcoeffs[i[0]] * gammatil,
            np.array([nutil[:, i[s]] for s in range(1, dim)]).T]))

    return V1


def eFvec(elem, m=2):

    if m == elem.ntransitions:
        return elem.eF

    elif m < elem.ntransitions:
        F1 = elem.F1[:m]
        return F1 / np.sqrt(F1 @ F1)


def Pparallel(elem, n=3, m=2):

    eF = eFvec(elem)
    Pparallel = np.kron(np.eye(n), np.outer(eF, eF))

    return Pparallel


def Pperp(elem, n=3, m=2):

    eF = eFvec(elem)
    Pperp = np.kron(np.eye(n), np.eye(m) - np.outer(eF, eF))

    return Pperp


def SigmalphaNP(elem, nsamples, dim=3, detstr="gkp"):

    sa, sa_perp, sa_para = [], [], []

    for a_inds, i_inds in product(combinations(elem.range_a, dim),
            combinations(elem.range_i, dim - 1)):

        nutil, Xcoeffs, gammatil, mutil = prepare_dataset(elem, a_inds, i_inds)

        Sigma = covnutil_ai(elem, a_inds, i_inds, nsamples)
        alpha_grad = grad_alphaNP_nutil(nutil, Xcoeffs, gammatil, mutil)

        P_perp = Pperp(elem)
        P_para = Pparallel(elem)

        SigmalphaNP = alpha_grad @ Sigma @ alpha_grad.T
        SigmalphaNP_perp = alpha_grad @ P_perp @ Sigma @ P_perp @ alpha_grad.T
        SigmalphaNP_parallel = alpha_grad @ P_para @ Sigma @ P_para @ alpha_grad.T

        sa.append(SigmalphaNP)
        sa_perp.append(SigmalphaNP_perp)
        sa_para.append(SigmalphaNP_parallel)

    return np.array(sa), np.array(sa_perp), np.array(sa_para)
