import numpy as np
import logging
import os
import json
from scipy.linalg import cho_factor, cho_solve

np.set_printoptions(precision=36, floatmode='maxprec')

from kifit.build import Elem
from kifit.fitools import (
    generate_elemsamples, generate_alphaNP_samples,
    get_llist_elemsamples, get_delchisq, get_delchisq_crit, get_confint,
    choLL, perform_polyfit)
from kifit.detools import (sample_gkp_combinations, sample_proj_combinations,
                           generate_alphaNP_dets, )

from Mathematica_crosschecks import(
        muvec_Mathematica,
        kappaperp1nit_LL_Mathematica,
        ph1nit_LL_Mathematica,
        D_a1i_Mathematica,
        dmat_Mathematica,
        dmat_Mathematica_min,
        absd_Mathematica,
        absd_Mathematica_min,
        absd_alpha1em11_Mathematica_min,
        D_a1i_1_Mathematica,
        D_a1i_1_alpha1em11_Mathematica,
        ph1_2_Mathematica,
        ph1_2_Mathematica_red,
        D_a1i_2_Mathematica,
        D_a1i_2_alpha1em11_Mathematica,
        ph1_3_Mathematica,
        ph1_3_Mathematica_red,
        D_a1i_3_Mathematica,
        ph1_4_Mathematica,
        D_a1i_4_Mathematica,
        covd_Camintest_N1e2,
        covd_Camintest_N1e3,
        covd_Camintest_N1e4,
        covd_Camintest_N1e5,
        covd_Camintest_alpha1em11_N1e2,
        covd_Camintest_alpha1em11_N1e3,
        covd_Camintest_alpha1em11_N1e4,
        covd_Camintest_alpha1em11_N1e5,
        covd_Camintest_kifitparams_N1e2,
        covd_Camintest_kifitparams_N1e3,
        covd_Camintest_kifitparams_N1e4,
        covd_Camintest_kifitparams_N1e5,
        covd_Camintest_kifitparams_alpha1em11_N1e2,
        covd_Camintest_kifitparams_alpha1em11_N1e3,
        covd_Camintest_kifitparams_alpha1em11_N1e4,
        covd_Camintest_kifitparams_alpha1em11_N1e5,
        )

np.random.seed(1)
fsize = 12
axislabelsize = 15

inputfolder = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "test_input"))
if not os.path.exists(inputfolder):
    logging.info("Creating file at ", inputfolder)
    os.mkdir(inputfolder)

outputfolder = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "test_output"))
if not os.path.exists(outputfolder):
    logging.info("Creating file at ", outputfolder)
    os.mkdir(outputfolder)


def get_det_vals(elem, nsamples, dim, method_tag):

    alphas, sigalphas, num_perm = generate_alphaNP_dets(
            elem=elem,
            nsamples=nsamples,
            dim=dim,
            detstr=method_tag)

    UB = alphas + 2 * sigalphas
    LB = alphas - 2 * sigalphas

    return (alphas, LB, UB)


def swap_inputparams(inputparams, nisotopepairs, ntransitions):

    inputparamat = np.reshape(inputparams[2 * nisotopepairs:],
        (nisotopepairs, ntransitions)).T

    inputparamat_swap = np.c_[inputparamat[1], inputparamat[0]]

    inputparamswap = np.concatenate((inputparams[: 2 * nisotopepairs],
        inputparamat_swap), axis=None)

    return inputparamswap

def swap_dmat(dmat):

    swap_dmat = (dmat).T

    return np.c_[swap_dmat[1], swap_dmat[0]]

def compute_covmats(elem, inputparamsamples, fitparamsamples, symm=True):

    nutilsamples = []
    absdsamples = []

    simpDeltamatsamples = []
    Deltamatsamples = []
    dmatsamples = []

    for i, inputparams in enumerate(inputparamsamples):

        elem._update_elem_params(inputparams)
        fitparams = fitparamsamples[i]
        elem._update_fit_params(fitparams)

        nutilsamples.append(elem.nutil.flatten())
        absdsamples.append(elem.absd(symm))

        simpDeltamatsamples.append(np.array([[
            elem.D_a1i(a, i, True)
            for i in elem.range_i] for a in elem.range_a]))

        Deltamatsamples.append(np.array([[
            elem.nutil[a, i]
            - elem.F1[i] * elem.nutil[a, 0]
            - elem.secph1[i] * elem.Kperp1[i]
            for i in elem.range_i] for a in elem.range_a]))
        dmatsamples.append(elem.dmat(symm))

    nutilcovmat_flattened = np.cov(np.array(nutilsamples), rowvar=False)
    dcovmat = np.cov(np.array(absdsamples), rowvar=False)

    return (nutilcovmat_flattened, dcovmat, absdsamples,
            np.array(simpDeltamatsamples),
            np.array(Deltamatsamples),
            np.array(dmatsamples))


def compute_inverse(mat):
    """
    Compute the inverse of the matrix mat using the Cholesky decomposition.
    Assumes mat is symmetric, positive definite.
    """

    cho_factorised = cho_factor(mat, lower=True)
    matinv = cho_solve(cho_factorised, np.eye(mat.shape[0]))

    return matinv


def compute_spectral_difference(Amat, Bmat):

    evals_A, evecs_A = np.linalg.eig(Amat)
    evals_B, evecs_B = np.linalg.eig(Bmat)

    assert all(evals_A > 0)
    assert all(evals_B > 0)

    eval_diff = np.sum(np.abs(evals_A - evals_B)) / np.sum(evals_A)

    return eval_diff

def Frobenius_norm(Mat):

    return np.sqrt(np.sum(Mat**2))

def compute_Frobenius_norm_difference(Amat, Bmat):

    Frob_diff = Frobenius_norm(Amat - Bmat) / Frobenius_norm(Amat)

    return Frob_diff


def compute_Kullback_Leibler_divergence(Amat, Bmat):

    covnutil_KL_div = 1 / 2 * (
            np.trace(compute_inverse(Bmat) @ Amat)
            - Amat.shape[0]
            + np.log(np.linalg.det(Bmat) / np.linalg.det(Amat)))

    return covnutil_KL_div


def test_sample_woNP():

    ca = Elem('Ca_testdata')

    orig_fitparams = np.array([ca.kp1_init, ca.ph1_init])
    sig_orig_fitparams = np.array([ca.sig_kp1_init, ca.sig_ph1_init])

    D_a1i_python_orig_fitparams = [[
        ca.D_a1i(a, i, True) for i in ca.range_i] for a in ca.range_a]

    dmat_symm_python_orig_fitparams = ca.dmat(True)
    absd_symm_python_orig_fitparams = ca.absd(True)

    # overwrite nuclear masses with atomic masses
    ca._update_elem_params(np.concatenate((
        ca.isotope_data[1],
        ca.isotope_data[4],
        (ca.nu_in).flatten())))

    assert np.allclose(ca.muvec, muvec_Mathematica, atol=0, rtol=1e-10)

    theta_LL_Mathematica = np.concatenate((kappaperp1nit_LL_Mathematica,
                                        ph1nit_LL_Mathematica, 0.), axis=None)

    ca._update_fit_params(theta_LL_Mathematica)

    new_fitparams = np.array([ca.kp1, ca.ph1])

    # the difference between the fitparams (kperp, phi) determined by
    # Mathematica & kifit are within the uncertainties claimed by kifit
    assert (np.abs(new_fitparams - orig_fitparams) < sig_orig_fitparams).all()

    D_a1i_python = [[ca.D_a1i(a, i, True) for i in ca.range_i] for a in ca.range_a]

    assert np.allclose(D_a1i_Mathematica, D_a1i_python, atol=0, rtol=1e-14)
    assert np.allclose(np.array(dmat_Mathematica) / ca.dnorm, ca.dmat(True),
        atol=0, rtol=1e-8)

    # with (kperp, phi) as determined by kifit, rtol=1e-2 compared to kifit result
    # with (kperp, phi) as determined by Mathematica, whereas for certain
    # elements in dmat the difference is even larger.
    # with same (kperp, phi) as in Mathematica, rtol=1e-9
    assert np.allclose(ca.dmat(True), dmat_symm_python_orig_fitparams,
                       atol=0, rtol=2)
    assert np.allclose(ca.absd(True), absd_symm_python_orig_fitparams,
                       atol=0, rtol=1e-2)
    assert np.allclose(ca.absd(True), absd_Mathematica,
                       atol=0, rtol=1e-9)

    ##############################

    camin = Elem('Camin_testdata')

    # print("sigrel_kp1_init", np.abs(camin.sig_kp1_init / camin.kp1_init))
    # print("sigrel_ph1_init", np.abs(camin.sig_ph1_init / camin.ph1_init))

    camin._update_elem_params(np.concatenate((
        camin.isotope_data[1],
        camin.isotope_data[4],
        (camin.nu_in).flatten())))

    assert np.allclose(camin.muvec, muvec_Mathematica, atol=0, rtol=1e-10)

    dmat_symm_python_orig_fitparams_min = camin.dmat(True)
    absd_symm_python_orig_fitparams_min = camin.absd(True)

    theta_Mathematica_min = np.array([
        kappaperp1nit_LL_Mathematica[0], ph1nit_LL_Mathematica[0], 0.])

    camin._update_fit_params(theta_Mathematica_min)

    # for Camin_testdata, just first two transitions
    D_a1i_python_min = np.array([[camin.D_a1i(a, i, True)
                         for i in camin.range_i] for a in camin.range_a])

    D_a1i_Mathematica_min = [[da1i[0], da1i[1]] for da1i in D_a1i_Mathematica]

    assert np.allclose(D_a1i_Mathematica_min, D_a1i_python_min,
                       atol=0, rtol=1e-13)
    assert np.allclose(D_a1i_1_Mathematica, D_a1i_python_min,
                       atol=0, rtol=1e-13)

    KF_5_Mathematica = np.array(np.genfromtxt(
        os.path.join(inputfolder,
                     "lLopt_fitparams_Ca_testdata.txt")))

    camin._update_fit_params(np.append(KF_5_Mathematica, np.array([0.])))

    D_a1i_python_load = np.array([[camin.D_a1i(a, i, True)
                         for i in camin.range_i] for a in camin.range_a])

    assert np.allclose(D_a1i_1_Mathematica, D_a1i_python_load,
                       atol=0, rtol=1e-13)

    camin._update_fit_params(np.array([
        kappaperp1nit_LL_Mathematica[0], ph1_2_Mathematica, 0.]))

    D_a1i_2_python = np.array([[camin.D_a1i(a, i, True)
                         for i in camin.range_i] for a in camin.range_a])

    assert np.allclose(D_a1i_2_Mathematica, D_a1i_2_python,
                       atol=0, rtol=1e-14)
    assert np.allclose(D_a1i_2_python, D_a1i_2_Mathematica,
                       atol=0, rtol=1e-14)

    camin._update_fit_params(np.array([
        kappaperp1nit_LL_Mathematica[0], ph1_2_Mathematica_red, 0.]))

    D_a1i_2_python_red = np.array([[camin.D_a1i(a, i, True)
                         for i in camin.range_i] for a in camin.range_a])

    assert np.allclose(D_a1i_2_Mathematica, D_a1i_2_python_red,
                       atol=0, rtol=1e-14)
    assert np.allclose(D_a1i_2_python_red, D_a1i_2_Mathematica,
                       atol=0, rtol=1e-13)


    camin._update_fit_params(np.array([
        kappaperp1nit_LL_Mathematica[0], ph1_3_Mathematica, 0.]))

    D_a1i_3_python = np.array([[camin.D_a1i(a, i, True)
                         for i in camin.range_i] for a in camin.range_a])

    assert np.allclose(D_a1i_3_Mathematica, D_a1i_3_python,
                       atol=0, rtol=1e-14)
    assert np.allclose(D_a1i_3_python, D_a1i_3_Mathematica,
                       atol=0, rtol=1e-14)

    camin._update_fit_params(np.array([
        kappaperp1nit_LL_Mathematica[0], ph1_3_Mathematica_red, 0.]))

    D_a1i_3_python_red = np.array([[camin.D_a1i(a, i, True)
                         for i in camin.range_i] for a in camin.range_a])

    assert np.allclose(D_a1i_3_Mathematica, D_a1i_3_python_red,
                       atol=0, rtol=1e-14)
    assert np.allclose(D_a1i_3_python_red, D_a1i_3_Mathematica,
                       atol=0, rtol=1e-14)


    fitparams_4_Mathematica = np.array([
        kappaperp1nit_LL_Mathematica[0], ph1_4_Mathematica, 0.])

    camin._update_fit_params(fitparams_4_Mathematica)

    D_a1i_4_python = np.array([[camin.D_a1i(a, i, True)
                         for i in camin.range_i] for a in camin.range_a])

    assert np.allclose(D_a1i_4_Mathematica, D_a1i_4_python,
                       atol=0, rtol=1e-14)
    assert np.allclose(D_a1i_4_python, D_a1i_4_Mathematica,
                       atol=0, rtol=1e-14)


    camin._update_fit_params(np.append(KF_5_Mathematica, np.array([0.])))

    assert np.allclose(np.array(dmat_Mathematica_min) / camin.dnorm,
                       camin.dmat(True),
                       atol=0, rtol=1e-8)

    assert np.allclose(camin.dmat(True), dmat_symm_python_orig_fitparams_min,
                       atol=0, rtol=2)
    assert np.allclose(camin.absd(True), absd_symm_python_orig_fitparams_min,
                       atol=0, rtol=1)
    assert np.allclose(camin.absd(True), absd_Mathematica_min,
                       atol=0, rtol=1e-9)



def test_sample_wNP():

    camin = Elem('Camin_testdata')
    alphaval = 1e-11

    # overwrite nuclear masses with atomic masses
    camin._update_elem_params(np.concatenate((
        camin.isotope_data[1],
        camin.isotope_data[4],
        (camin.nu_in).flatten())))

    assert np.allclose(camin.muvec, muvec_Mathematica, atol=0, rtol=1e-10)

    camin._update_fit_params(np.array([
        kappaperp1nit_LL_Mathematica[0], ph1nit_LL_Mathematica[0], alphaval]))

    D_a1i_1_alpha1em11_python = np.array([[
        camin.D_a1i(a, i, True)
        for i in camin.range_i] for a in camin.range_a])

    assert np.allclose(D_a1i_1_alpha1em11_Mathematica,
                       D_a1i_1_alpha1em11_python,
                       atol=0, rtol=1e-14)

    camin._update_fit_params(np.array([
        kappaperp1nit_LL_Mathematica[0], ph1_2_Mathematica, alphaval]))

    D_a1i_2_alpha1em11_python = np.array([[
        camin.D_a1i(a, i, True)
        for i in camin.range_i] for a in camin.range_a])

    assert np.allclose(D_a1i_2_alpha1em11_Mathematica,
                       D_a1i_2_alpha1em11_python,
                       atol=0, rtol=1e-14)

    KF_5_Mathematica = np.array(np.genfromtxt(
        os.path.join(inputfolder,
                     "lLopt_fitparams_Ca_testdata.txt")))

    camin._update_fit_params(np.append(KF_5_Mathematica, np.array([1e-11])))

    assert np.allclose(camin.absd(True), absd_alpha1em11_Mathematica_min,
                       atol=0, rtol=1e-11)



def test_covmats():

    camin = Elem('Camin_testdata')

    # reconstruct covariance matrices from samples

    nusamples_Mathematica = np.genfromtxt(
            os.path.join(inputfolder, "nusamples_Ca_testdata.txt"),
            delimiter=",")

    mpsamples_Mathematica = np.genfromtxt(
            os.path.join(inputfolder, "mpsamples_Ca_testdata.txt"),
            delimiter=",")

    msamples_Mathematica = np.genfromtxt(
            os.path.join(inputfolder, "msamples_Ca_testdata.txt"),
            delimiter=",")

    inputparamsamples_Mathematica = np.c_[
            msamples_Mathematica, msamples_Mathematica, msamples_Mathematica,
            mpsamples_Mathematica,
            nusamples_Mathematica]

    covnutilmat_Mathematica = np.genfromtxt(
            os.path.join(inputfolder, "covnutilmat_Ca_testdata.txt"),
            delimiter=",")

    fitparams_Mathematica = np.genfromtxt(
            os.path.join(inputfolder,
                         "lLopt_fitparams_Ca_testdata.txt"))

    fitparams_Mathematica_arrayed = np.repeat(
            np.array([fitparams_Mathematica.tolist() + [0]]),
            inputparamsamples_Mathematica.shape[0],
            axis=0)

    simpDeltamatsamples_elemvar_Mathematica = np.genfromtxt(
            os.path.join(inputfolder,
                         "simpDeltamatsamples_elemvar_Ca_testdata.txt"),
            delimiter=",").reshape(
                    nusamples_Mathematica.shape[0],
                    camin.nisotopepairs,
                    camin.ntransitions)

    Deltamatsamples_elemvar_Mathematica = np.genfromtxt(
        os.path.join(inputfolder,
                     "Deltamatsamples_elemvar_Ca_testdata.txt"),
        delimiter=",").reshape(
                nusamples_Mathematica.shape[0],
                camin.nisotopepairs,
                camin.ntransitions)

    dmatsamples_elemvar_Mathematica = np.genfromtxt(
        os.path.join(inputfolder,
                     "dmatsamples_elemvar_Ca_testdata.txt"),
        delimiter=",").reshape(nusamples_Mathematica.shape[0],
                               camin.nisotopepairs,
                               camin.ntransitions)

    absdsamples_elemvar_Mathematica = np.genfromtxt(
            os.path.join(inputfolder,
                         "absdsamples_elemvar_Ca_testdata.txt"),
            delimiter=",")

    covdmat_absd_elemvar_Mathematica_64 = np.cov(
            absdsamples_elemvar_Mathematica, rowvar=False)

    covdmat_absd_elemvar_Mathematica_128 = np.cov(
            absdsamples_elemvar_Mathematica.astype(np.float128),
            rowvar=False, dtype=np.float128)

    covdmat_elemvar_Mathematica = np.genfromtxt(
            os.path.join(
                inputfolder,
                "covdmat_elemvar_Ca_testdata.txt"),
            delimiter=",")

    logLsamples_elemvar_Mathematica = np.genfromtxt(
            os.path.join(
                inputfolder,
                "logLsamples_elemvar_Ca_testdata.txt"),
            delimiter=",")

    (
        covnutilmat_elemvar_Math2kifit,
        covdmat_elemvar_Math2kifit,
        absdsamples_elemvar_Math2kifit,
        simpDeltamatsamples_elemvar_Math2kifit,
        Deltamatsamples_elemvar_Math2kifit,
        dmatsamples_elemvar_Math2kifit) = compute_covmats(
            camin,
            inputparamsamples_Mathematica,
            fitparams_Mathematica_arrayed,
            symm=True)

    assert np.allclose(covnutilmat_Mathematica,
                       covnutilmat_elemvar_Math2kifit,
                       atol=0, rtol=1e-5)
    assert np.allclose(covnutilmat_elemvar_Math2kifit,
                       covnutilmat_Mathematica,
                       atol=0, rtol=1e-5)
    assert np.allclose(simpDeltamatsamples_elemvar_Mathematica[0],
                       simpDeltamatsamples_elemvar_Math2kifit[0],
                       atol=0, rtol=1e-14)
    assert np.allclose(Deltamatsamples_elemvar_Mathematica,
                       Deltamatsamples_elemvar_Math2kifit,
                       atol=0, rtol=1e-6)
    assert np.allclose(dmatsamples_elemvar_Mathematica,
                       dmatsamples_elemvar_Math2kifit,
                       atol=0, rtol=1e-6)
    assert np.allclose(absdsamples_elemvar_Mathematica,
                       absdsamples_elemvar_Math2kifit,
                       atol=0, rtol=1e-6)
    assert np.allclose(absdsamples_elemvar_Math2kifit,
                       absdsamples_elemvar_Mathematica,
                       atol=0, rtol=1e-6)
    assert np.allclose(np.mean(absdsamples_elemvar_Mathematica, axis=0),
                       np.mean(absdsamples_elemvar_Math2kifit, axis=0),
                       atol=0, rtol=1e-10)
    assert np.allclose(np.mean(absdsamples_elemvar_Math2kifit, axis=0),
                       np.mean(absdsamples_elemvar_Mathematica, axis=0),
                       atol=0, rtol=1e-10)
    assert np.allclose(np.max(absdsamples_elemvar_Mathematica, axis=0),
                       np.max(absdsamples_elemvar_Math2kifit, axis=0),
                       atol=0, rtol=1e-11)
    assert np.allclose(np.mean(absdsamples_elemvar_Math2kifit, axis=0),
                       np.mean(absdsamples_elemvar_Mathematica, axis=0),
                       atol=0, rtol=1e-10)
    assert np.allclose(covdmat_elemvar_Mathematica,
                       covdmat_elemvar_Math2kifit,
                       atol=0, rtol=1e-8)
    assert np.allclose(covdmat_elemvar_Math2kifit,
                       covdmat_elemvar_Mathematica,
                       atol=0, rtol=1e-8)
    assert np.allclose(covdmat_absd_elemvar_Mathematica_64,
                       covdmat_elemvar_Math2kifit,
                       atol=0, rtol=1e-9)
    assert np.allclose(covdmat_absd_elemvar_Mathematica_128,
                       covdmat_elemvar_Math2kifit,
                       atol=0, rtol=1e-9)
    assert np.allclose(covdmat_absd_elemvar_Mathematica_128,
                       covdmat_absd_elemvar_Mathematica_64,
                       atol=0, rtol=1e-13)

    ll_elemvar_Mathematica = np.array([
        choLL(absd_elemvar_Mathematica, covdmat_elemvar_Mathematica)
        for absd_elemvar_Mathematica in absdsamples_elemvar_Mathematica])

    ll_elemvar_Math2kifit = np.array([
        choLL(absd_elemvar_Math2kifit, covdmat_elemvar_Math2kifit)
        for absd_elemvar_Math2kifit in absdsamples_elemvar_Math2kifit])

    assert np.allclose(ll_elemvar_Mathematica, ll_elemvar_Math2kifit,
                       atol=0, rtol=1e-10)
    assert np.isclose(np.min(ll_elemvar_Mathematica),
                      np.min(ll_elemvar_Math2kifit),
                      atol=0, rtol=1e-11)
    assert np.isclose(np.max(ll_elemvar_Mathematica),
                      np.max(ll_elemvar_Math2kifit),
                      atol=0, rtol=1e-11)
    assert np.allclose(logLsamples_elemvar_Mathematica, ll_elemvar_Mathematica,
                       atol=0, rtol=1e-15)
    assert np.allclose(logLsamples_elemvar_Mathematica, ll_elemvar_Math2kifit,
                       atol=0, rtol=1e-10)

    # with F, K sampling

    F21samples_Mathematica = np.genfromtxt(
            os.path.join(inputfolder,"F21samples_Ca_testdata.txt"),
            delimiter=",")
    ph21samples_Mathematica = np.arctan(F21samples_Mathematica)

    K21samples_Mathematica = np.genfromtxt(
            os.path.join(inputfolder, "K21samples_Ca_testdata.txt"),
            delimiter=",")
    Kperp21samples_Mathematica = (
            K21samples_Mathematica * np.cos(ph21samples_Mathematica))

    fitparamsamples_Mathematica = np.c_[Kperp21samples_Mathematica,
                                        ph21samples_Mathematica,
                                        np.zeros(len(Kperp21samples_Mathematica))]

    simpDeltamatsamples_fitparamvar_Mathematica = np.genfromtxt(
            os.path.join(inputfolder,
                         "simpDeltamatsamples_fitparamvar_Ca_testdata.txt"),
            delimiter=",").reshape(nusamples_Mathematica.shape[0],
                                   camin.nisotopepairs,
                                   camin.ntransitions)

    Deltamatsamples_fitparamvar_Mathematica = np.genfromtxt(
            os.path.join(inputfolder,
                         "Deltamatsamples_fitparamvar_Ca_testdata.txt"),
            delimiter=",").reshape(nusamples_Mathematica.shape[0],
                                   camin.nisotopepairs,
                                   camin.ntransitions)

    dmatsamples_fitparamvar_Mathematica = np.genfromtxt(
            os.path.join(inputfolder,
                         "dmatsamples_fitparamvar_Ca_testdata.txt"),
            delimiter=",").reshape(nusamples_Mathematica.shape[0],
                                   camin.nisotopepairs,
                                   camin.ntransitions)

    absdsamples_fitparamvar_Mathematica = np.genfromtxt(
        os.path.join(inputfolder,
                     "absdsamples_fitparamvar_Ca_testdata.txt"),
        delimiter=",")

    covdmat_fitparamvar_Mathematica = np.genfromtxt(
        os.path.join(inputfolder,
                     "covdmat_fitparamvar_Ca_testdata.txt"),
        delimiter=",")

    logLsamples_fitparamvar_Mathematica = np.genfromtxt(
        os.path.join(inputfolder,
                     "logLsamples_fitparamvar_Ca_testdata.txt"),
        delimiter=",")

    (
            covnutilmat_fitparamvar_Math2kifit,
            covdmat_fitparamvar_Math2kifit,
            absdsamples_fitparamvar_Math2kifit,
            simpDeltamatsamples_fitparamvar_Math2kifit,
            Deltamatsamples_fitparamvar_Math2kifit,
            dmatsamples_fitparamvar_Math2kifit) = compute_covmats(
                    camin,
                    inputparamsamples_Mathematica,
                    fitparamsamples_Mathematica,
                    symm=True)

    assert np.allclose(covnutilmat_Mathematica,
                       covnutilmat_fitparamvar_Math2kifit,
                       atol=0, rtol=1e-5)
    assert np.allclose(simpDeltamatsamples_fitparamvar_Mathematica,
                       simpDeltamatsamples_fitparamvar_Math2kifit,
                       atol=0, rtol=1e-12)
    assert np.allclose(Deltamatsamples_fitparamvar_Mathematica,
                       Deltamatsamples_fitparamvar_Math2kifit,
                       atol=0, rtol=1e-9)
    assert np.allclose(dmatsamples_fitparamvar_Mathematica,
                       dmatsamples_fitparamvar_Math2kifit,
                       atol=0, rtol=1e-9)
    assert np.allclose(absdsamples_fitparamvar_Mathematica,
                       absdsamples_fitparamvar_Math2kifit,
                       atol=0, rtol=1e-9)
    assert np.allclose(covdmat_fitparamvar_Mathematica,
                       covdmat_fitparamvar_Math2kifit,
                       atol=0, rtol=1e-11)

    ll_fitparamvar_Mathematica = np.array([
        choLL(absd_fitparamvar_Mathematica, covdmat_fitparamvar_Mathematica)
        for absd_fitparamvar_Mathematica in absdsamples_fitparamvar_Mathematica])

    ll_fitparamvar_Math2kifit = np.array([
        choLL(absd_fitparamvar_Math2kifit, covdmat_fitparamvar_Math2kifit)
        for absd_fitparamvar_Math2kifit in absdsamples_fitparamvar_Math2kifit])

    assert np.allclose(ll_fitparamvar_Mathematica, ll_fitparamvar_Math2kifit,
                       atol=0, rtol=1e-5)
    assert np.allclose(ll_fitparamvar_Math2kifit, ll_fitparamvar_Mathematica,
                       atol=0, rtol=1e-5)
    assert np.allclose(logLsamples_fitparamvar_Mathematica,
                       ll_fitparamvar_Mathematica,
                       atol=0, rtol=1e-7)
    assert np.allclose(logLsamples_fitparamvar_Mathematica,
                       ll_fitparamvar_Mathematica,
                       atol=0, rtol=1e-7)
    assert np.isclose(np.min(ll_fitparamvar_Mathematica),
                      np.min(ll_fitparamvar_Math2kifit),
                      atol=0, rtol=1e-7)
    assert np.isclose(np.max(ll_fitparamvar_Mathematica),
                      np.max(ll_fitparamvar_Math2kifit),
                      atol=0, rtol=1e-5)

def test_sampling():

    camin = Elem('Camin_testdata')

    Nsamples = [int(1e2), int(1e3), int(1e4), int(1e5)]

    # loglikelihood-optimised fitparams
    covdmats_Mathematica = np.array([
            covd_Camintest_kifitparams_N1e2,
            covd_Camintest_kifitparams_N1e3,
            covd_Camintest_kifitparams_N1e4,
            covd_Camintest_kifitparams_N1e5])

    covdmats_alpha1em11_Mathematica = np.array([
            covd_Camintest_kifitparams_alpha1em11_N1e2,
            covd_Camintest_kifitparams_alpha1em11_N1e3,
            covd_Camintest_kifitparams_alpha1em11_N1e4,
            covd_Camintest_kifitparams_alpha1em11_N1e5])

    logLsamples_N1e2 = np.genfromtxt(
            os.path.join(inputfolder,
                         "logLsamples_fitparamvar_Ca_testdata_Ns100.txt"),
            delimiter=",")
    logLsamples_N1e3 = np.genfromtxt(
            os.path.join(inputfolder,
                         "logLsamples_fitparamvar_Ca_testdata_Ns1000.txt"),
            delimiter=",")
    logLsamples_N1e4 = np.genfromtxt(
            os.path.join(inputfolder,
                         "logLsamples_fitparamvar_Ca_testdata_Ns10000.txt"),
            delimiter=",")
    logLsamples_N1e5 = np.genfromtxt(
            os.path.join(inputfolder,
                         "logLsamples_fitparamvar_Ca_testdata_Ns100000.txt"),
            delimiter=",")

    logLsamples_Mathematica = [
        logLsamples_N1e2, logLsamples_N1e3, logLsamples_N1e4, logLsamples_N1e5]


    logLsamples_alpha1em11_N1e2 = np.genfromtxt(
        os.path.join(inputfolder,
                     "logLsamples_fitparamvar_alpha1em11_Ca_testdata_Ns100.txt"),
        delimiter=",")
    logLsamples_alpha1em11_N1e3 = np.genfromtxt(
        os.path.join(inputfolder,
                     "logLsamples_fitparamvar_alpha1em11_Ca_testdata_Ns1000.txt"),
        delimiter=",")
    logLsamples_alpha1em11_N1e4 = np.genfromtxt(
        os.path.join(inputfolder,
                     "logLsamples_fitparamvar_alpha1em11_Ca_testdata_Ns10000.txt"),
        delimiter=",")
    logLsamples_alpha1em11_N1e5 = np.genfromtxt(
        os.path.join(inputfolder,
                     "logLsamples_fitparamvar_alpha1em11_Ca_testdata_Ns100000.txt"),
        delimiter=",")

    logLsamples_alpha1em11_Mathematica = [
        logLsamples_alpha1em11_N1e2,
        logLsamples_alpha1em11_N1e3,
        logLsamples_alpha1em11_N1e4,
        logLsamples_alpha1em11_N1e5]

    covnutil_spec = []
    covnutil_Frob = []
    covnutil_KL = []

    covd_spec = []
    covd_Frob = []
    covd_KL = []

    for n, Ns in enumerate(Nsamples):

        inputparamsamples, fitparamsamples = generate_elemsamples(camin, Ns)

        np.savetxt(os.path.join(outputfolder,
                                f"{camin.id}_fitparams_Ns{Ns}.txt"),
                   fitparamsamples, delimiter=",")

        fitparams = np.c_[fitparamsamples, np.zeros(Ns)]

        (
                covnutilmat_kifit,
                covdmat_kifit,
                absdsamples_kifit,
                simpDeltamatsamples_kifit,
                Deltamatsamples_kifit,
                dmatsamples_kift) = compute_covmats(camin,
                                                    inputparamsamples,
                                                    fitparams,
                                                    symm=True)

        logLsamples_kifit = np.array([
            choLL(absdsample, covdmat_kifit) for absdsample in absdsamples_kifit])

        fitparams_alpha1em11 = np.c_[fitparamsamples, 1e-11 * np.ones(Ns)]

        (
                covnutilmat_alpha1em11_kifit,
                covdmat_alpha1em11_kifit,
                absdsamples_alpha1em11_kifit,
                simpDeltamatsamples_alpha1em11_kifit,
                Deltamatsamples_alpha1em11_kifit,
                dmatsamples_alpha1em11_kift) = compute_covmats(camin,
                                                               inputparamsamples,
                                                               fitparams_alpha1em11,
                                                               symm=True)

        logLsamples_alpha1em11_kifit = np.array([
            choLL(absdsample, covdmat_alpha1em11_kifit)
            for absdsample in absdsamples_alpha1em11_kifit])


        assert np.allclose(covdmat_kifit,
                           covdmats_Mathematica[n],
                           atol=0, rtol=10)

        assert np.allclose(covdmat_alpha1em11_kifit,
                           covdmats_alpha1em11_Mathematica[n],
                           atol=0, rtol=10)


        # condition number
        assert (np.linalg.cond(covdmat_kifit) < 1e6)
        assert (np.linalg.cond(covdmats_Mathematica[n]) < 1e6)

        # spectrum
        covd_spec_diff = compute_spectral_difference(
                covdmats_Mathematica[n], covdmat_kifit)

        covd_spec.append(covd_spec_diff)

        # Frobenius Norm Difference
        covd_Frob_diff = compute_Frobenius_norm_difference(
                covdmats_Mathematica[n], covdmat_kifit)

        covd_Frob.append(covd_Frob_diff)


        # Kullback-Leibler divergence
        covd_KL_div = compute_Kullback_Leibler_divergence(
                covdmats_Mathematica[n], covdmat_kifit)

        covd_KL.append(covd_KL_div)


        # inverse
        covdmatinv_kifit = compute_inverse(covdmat_kifit)
        covdmatinv_Mathematica = compute_inverse(covdmats_Mathematica[n])

        assert (np.abs(np.max(
            (covdmatinv_kifit @ covdmat_kifit)
            - np.eye(covdmatinv_kifit.shape[0])
            )) < 1e-14)

        assert (np.abs(np.max(
            (covdmatinv_Mathematica @ covdmats_Mathematica[n])
             - np.eye(covdmatinv_Mathematica.shape[0])
            )) < 1e-14)

        # ll

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        plt.hist(logLsamples_kifit,
                 histtype="step", color="darkgreen", label="kifit w/o NP")
        plt.hist(logLsamples_Mathematica[n],
                 histtype="step", color="blue", label="Mathematica w/o NP")

        plt.hist(logLsamples_alpha1em11_kifit
                 - np.min(logLsamples_alpha1em11_kifit),
                 histtype="step", color="orange",
                 label=r"kifit, $\alpha_{\mathrm{NP}}=10^{-11}$")
        plt.hist(logLsamples_alpha1em11_Mathematica[n]
                 - np.min(logLsamples_alpha1em11_Mathematica[n]),
                 histtype="step", color="red",
                 label=r"Mathematica, $\alpha_{\mathrm{NP}}=10^{-11}$")

        plt.legend(loc="upper right", fontsize=fsize)

        ax.set_xlabel(r"$\log \mathcal{L}$", fontsize=axislabelsize)
        ax.set_ylabel("counts / " + str(Ns), fontsize=axislabelsize)

        plotpath = os.path.join(outputfolder,
                    f"{camin.id}_logLhist_x{camin.x}_symm_Ns{int(Ns)}.pdf")
        plt.savefig(plotpath)

        assert np.isclose(np.mean(logLsamples_kifit),
                          np.mean(logLsamples_Mathematica[n]),
                          atol=0, rtol=1e-1)
        assert np.isclose(np.mean(logLsamples_alpha1em11_kifit),
                          np.mean(logLsamples_alpha1em11_Mathematica[n]),
                          atol=0, rtol=1)

        assert np.isclose(np.min(logLsamples_kifit),
                          np.min(logLsamples_Mathematica[n]),
                          atol=0, rtol=1)
        assert np.isclose(np.min(logLsamples_kifit),
                          np.min(logLsamples_Mathematica[n]),
                          atol=0, rtol=1)


        assert np.isclose(
                np.mean(logLsamples_alpha1em11_kifit - logLsamples_kifit),
                np.mean(logLsamples_alpha1em11_Mathematica[n]
                        - logLsamples_Mathematica[n]),
                atol=0, rtol=1)


    covd_spec = np.array(covd_spec)
    covd_Frob = np.array(covd_Frob)
    covd_KL = np.array(covd_KL)


    assert (covd_spec < 0.3).all()
    assert (covd_spec[:-1] == sorted(covd_spec[:-1], reverse=True)).all()
    assert (covd_spec[-1] < 0.01).all()

    assert (covd_Frob < 0.3).all()
    assert (covd_Frob == sorted(covd_Frob, reverse=True)).all()
    assert (covd_Frob[-1] < 0.01).all()

    assert (covd_KL < 0.03).all()
    assert (covd_KL == sorted(covd_KL, reverse=True)).all()
    assert (covd_KL[-1] < 1e-4).all()



def test_d_swap_varying_inputparams():

    camin = Elem('Camin')
    camin_swap = Elem('Camin_swap')

    swapped_swap_dmat = swap_dmat(camin_swap.dmat(True))
    assert np.allclose(swapped_swap_dmat[0], camin.dmat(True)[0], atol=0,
                       rtol=1e-4)
    assert np.allclose(swapped_swap_dmat[1], camin.dmat(True)[1], atol=0,
                       rtol=1e-4)
    assert np.allclose(swapped_swap_dmat[2], camin.dmat(True)[2], atol=0,
                       rtol=1e-4)

    swapped_swap_dmat = swap_dmat(camin_swap.dmat(False))
    assert np.allclose(swapped_swap_dmat[0], camin.dmat(False)[0], atol=0,
                       rtol=1e-4)
    assert np.allclose(swapped_swap_dmat[1], camin.dmat(False)[1], atol=0,
                       rtol=1e-3)
    assert np.allclose(swapped_swap_dmat[2], camin.dmat(False)[2], atol=0,
                       rtol=1e-4)

    assert np.isclose(camin.dnorm, camin_swap.dnorm, atol=0, rtol=1e-100)

    inputparamsamples, _ = generate_elemsamples(camin, 5)

    # alphaNP = 0
    ###########################################################################

    camin_absdsamples_alpha0 = []
    camin_swap_absdsamples_alpha0 = []

    for i, inputparams in enumerate(inputparamsamples):

        inputparamswap = swap_inputparams(inputparams,
                                          camin.nisotopepairs,
                                          camin.ntransitions)

        camin._update_elem_params(inputparams)
        camin_swap._update_elem_params(inputparamswap)

        swapped_swap_dmat = swap_dmat(camin_swap.dmat(True))
        assert np.allclose(swapped_swap_dmat[0], camin.dmat(True)[0], atol=0,
                           rtol=1e-3)
        assert np.allclose(swapped_swap_dmat[1], camin.dmat(True)[1], atol=0,
                           rtol=1e-3)
        assert np.allclose(swapped_swap_dmat[2], camin.dmat(True)[2], atol=0,
                           rtol=1e-2)

        swapped_swap_dmat = swap_dmat(camin_swap.dmat(False))
        assert np.allclose(swapped_swap_dmat[0], camin.dmat(False)[0], atol=0,
                           rtol=1e-4)
        assert np.allclose(swapped_swap_dmat[1], camin.dmat(False)[1], atol=0,
                           rtol=1e-3)
        assert np.allclose(swapped_swap_dmat[2], camin.dmat(False)[2], atol=0,
                           rtol=1e-2)


        assert np.isclose(camin.dnorm, camin_swap.dnorm, atol=0, rtol=1e-100)

        camin_absdsamples_alpha0.append(camin.absd(True))
        camin_swap_absdsamples_alpha0.append(camin_swap.absd(True))

    camin_covmat_alpha0 = np.cov(np.array(camin_absdsamples_alpha0),
                                 rowvar=False)
    camin_swap_covmat_alpha0 = np.cov(np.array(camin_swap_absdsamples_alpha0),
                                      rowvar=False)
    camin_mean_alpha0 = np.mean(np.array(camin_absdsamples_alpha0), axis=0)

    assert np.allclose(camin_covmat_alpha0, camin_swap_covmat_alpha0, atol=0,
                       rtol=1e-2)

    camin_ll_alpha0 = get_llist_elemsamples(camin_absdsamples_alpha0)
    camin_swap_ll_alpha0 = get_llist_elemsamples(camin_swap_absdsamples_alpha0)

    assert np.allclose(camin_ll_alpha0, camin_swap_ll_alpha0, atol=0, rtol=1e-2)

    # alphaNP = 1e-8
    ###########################################################################

    camin.alphaNP = 1e-8
    camin_swap.alphaNP = 1e-8

    camin_absdsamples_alpha1 = []
    camin_swap_absdsamples_alpha1 = []

    for i, inputparams in enumerate(inputparamsamples):

        inputparamswap = swap_inputparams(inputparams,
                                          camin.nisotopepairs,
                                          camin.ntransitions)

        camin._update_elem_params(inputparams)
        camin_swap._update_elem_params(inputparamswap)

        swapped_swap_dmat = swap_dmat(camin_swap.dmat(True))
        assert np.allclose(swapped_swap_dmat[0], camin.dmat(True)[0], atol=0,
                           rtol=1e-7)
        assert np.allclose(swapped_swap_dmat[1], camin.dmat(True)[1], atol=0,
                           rtol=1e-6)
        assert np.allclose(swapped_swap_dmat[2], camin.dmat(True)[2], atol=0,
                           rtol=1e-6)

        swapped_swap_dmat = swap_dmat(camin_swap.dmat(False))
        assert np.allclose(swapped_swap_dmat[0], camin.dmat(False)[0], atol=0,
                           rtol=1)
        assert np.allclose(swapped_swap_dmat[1], camin.dmat(False)[1], atol=0,
                           rtol=2)
        assert np.allclose(swapped_swap_dmat[2], camin.dmat(False)[2], atol=0,
                           rtol=1)

        assert np.isclose(camin.dnorm, camin_swap.dnorm, atol=0, rtol=1e-100)

        camin_absdsamples_alpha1.append(camin.absd(True))
        camin_swap_absdsamples_alpha1.append(camin_swap.absd(True))

    camin_covmat_alpha1 = np.cov(np.array(camin_absdsamples_alpha1),
                                 rowvar=False)
    camin_swap_covmat_alpha1 = np.cov(np.array(camin_swap_absdsamples_alpha1),
                                      rowvar=False)
    camin_mean_alpha1 = np.mean(np.array(camin_absdsamples_alpha1))

    assert np.allclose(camin_covmat_alpha1, camin_swap_covmat_alpha1,
        atol=0, rtol=1e-8)

    camin_ll_alpha1 = get_llist_elemsamples(camin_absdsamples_alpha1)
    camin_swap_ll_alpha1 = get_llist_elemsamples(camin_swap_absdsamples_alpha1)

    assert np.allclose(camin_ll_alpha1, camin_swap_ll_alpha1, atol=0, rtol=1e-7)

    # alphaNP = 1e-6
    ###########################################################################

    camin.alphaNP = 1e-6
    camin_swap.alphaNP = 1e-6

    camin_absdsamples_alpha2 = []
    camin_swap_absdsamples_alpha2 = []

    for i, inputparams in enumerate(inputparamsamples):

        inputparamswap = swap_inputparams(inputparams,
                                          camin.nisotopepairs,
                                          camin.ntransitions)

        camin._update_elem_params(inputparams)
        camin_swap._update_elem_params(inputparamswap)

        swapped_swap_dmat = swap_dmat(camin_swap.dmat(True))
        assert np.allclose(swapped_swap_dmat[0], camin.dmat(True)[0], atol=0,
                           rtol=1e-8)
        assert np.allclose(swapped_swap_dmat[1], camin.dmat(True)[1], atol=0,
                           rtol=1e-8)
        assert np.allclose(swapped_swap_dmat[2], camin.dmat(True)[2], atol=0,
                           rtol=1e-8)

        swapped_swap_dmat = swap_dmat(camin_swap.dmat(False))
        assert np.allclose(swapped_swap_dmat[0], camin.dmat(False)[0], atol=0,
                           rtol=1)
        assert np.allclose(swapped_swap_dmat[1], camin.dmat(False)[1], atol=0,
                           rtol=2)
        assert np.allclose(swapped_swap_dmat[2], camin.dmat(False)[2], atol=0,
                           rtol=1)

        assert np.isclose(camin.dnorm, camin_swap.dnorm, atol=0, rtol=1e-100)

        camin_absdsamples_alpha2.append(camin.absd(True))
        camin_swap_absdsamples_alpha2.append(camin_swap.absd(True))

    camin_covmat_alpha2 = np.cov(np.array(camin_absdsamples_alpha2),
                                 rowvar=False)
    camin_swap_covmat_alpha2 = np.cov(np.array(camin_swap_absdsamples_alpha2),
                                      rowvar=False)
    camin_mean_alpha2 = np.mean(np.array(camin_absdsamples_alpha2))

    assert np.allclose(camin_covmat_alpha2, camin_swap_covmat_alpha2,
        atol=0, rtol=1e-8)

    camin_ll_alpha2 = get_llist_elemsamples(camin_absdsamples_alpha2)
    camin_swap_ll_alpha2 = get_llist_elemsamples(camin_swap_absdsamples_alpha2)

    assert np.allclose(camin_ll_alpha2, camin_swap_ll_alpha2, atol=0, rtol=1e-8)


def test_alphaNP_dependence_covdd():

    camin = Elem('Camin')

    inputparamsamples, _ = generate_elemsamples(camin, 500)

    # alphaNP = 0
    ###########################################################################

    camin_absdsamples_alpha0 = []

    for i, inputparams in enumerate(inputparamsamples):

        camin._update_elem_params(inputparams)
        camin_absdsamples_alpha0.append(camin.absd(True))

    camin_covmat_part_ll_alpha0 = np.log(np.linalg.det(
        np.cov(np.array(camin_absdsamples_alpha0), rowvar=False))) / 2

    camin_ll_alpha0 = get_llist_elemsamples(camin_absdsamples_alpha0)


    # alphaNP = 1e-8
    ###########################################################################

    camin.alphaNP = 1e-8

    camin_absdsamples_alpha1 = []

    for i, inputparams in enumerate(inputparamsamples):

        camin._update_elem_params(inputparams)
        camin_absdsamples_alpha1.append(camin.absd(True))

    camin_covmat_part_ll_alpha1 = np.log(np.linalg.det(
        np.cov(np.array(camin_absdsamples_alpha1), rowvar=False))) / 2

    camin_ll_alpha1 = get_llist_elemsamples(camin_absdsamples_alpha1)

    # alphaNP = 1e-6
    ###########################################################################

    camin.alphaNP = 1e-6

    camin_absdsamples_alpha2 = []

    for i, inputparams in enumerate(inputparamsamples):

        camin._update_elem_params(inputparams)

        camin_absdsamples_alpha2.append(camin.absd(True))

    camin_covmat_part_ll_alpha2 = np.log(np.linalg.det(
        np.cov(np.array(camin_absdsamples_alpha2), rowvar=False))) / 2

    camin_ll_alpha2 = get_llist_elemsamples(camin_absdsamples_alpha2)

    ###########################################################################

    assert (camin_covmat_part_ll_alpha0 < camin_covmat_part_ll_alpha1)
    assert (camin_covmat_part_ll_alpha0 < camin_covmat_part_ll_alpha2)

    assert (np.mean(camin_ll_alpha0) < np.mean(camin_ll_alpha1))
    assert (np.mean(camin_ll_alpha0) < np.mean(camin_ll_alpha2))
    assert (np.mean(camin_ll_alpha1) < np.mean(camin_ll_alpha2))

    assert (camin_covmat_part_ll_alpha0 / np.mean(camin_ll_alpha0)
            > camin_covmat_part_ll_alpha1 / np.mean(camin_ll_alpha1))
    assert (camin_covmat_part_ll_alpha0 / np.mean(camin_ll_alpha0)
            > camin_covmat_part_ll_alpha2 / np.mean(camin_ll_alpha2))
    assert (camin_covmat_part_ll_alpha1 / np.mean(camin_ll_alpha1)
            > camin_covmat_part_ll_alpha2 / np.mean(camin_ll_alpha2))

    assert np.isclose(camin_covmat_part_ll_alpha0 / np.mean(camin_ll_alpha0),
                      0.8617380374400448,
                      atol=0, rtol=1)
    assert np.isclose(camin_covmat_part_ll_alpha1 / np.mean(camin_ll_alpha1),
                      0.00017522413438041855,
                      atol=0, rtol=1)
    assert np.isclose(camin_covmat_part_ll_alpha2 / np.mean(camin_ll_alpha2),
                      1.7461390752985816e-08,
                      atol=0, rtol=1)

def test_write_covdd(
        elemid="Camin_testdata",
        alphaval=0.,
        nelemsamples=1000,
        min_percentile=0,
        lam=0,
        symm=False):

    elem = Elem(elemid)

    inputparamsamples, fitparamsamples = generate_elemsamples(elem, nelemsamples)

    absdsamples = []

    for i, fitparams in enumerate(fitparamsamples):

        # update fitparams
        kp1 = fitparams[0]
        ph1 = fitparams[1]
        alphaNP = alphaval

        if np.isclose(ph1, 0., atol=0, rtol=1e-17):
            raise Exception("ph1 is too close to zero")

        elem._update_fit_params([kp1, ph1, alphaNP])

        # update inputparams
        inputparams = inputparamsamples[i]
        elem._update_elem_params(inputparams)
        absdsamples.append(elem.absd(symm))

    covdd = (np.cov(np.array(absdsamples), rowvar=False)
                   + 1e-17 * np.eye(elem.nisotopepairs))

    evals_covdd, evecs_covdd = np.linalg.eig(covdd)

    cond_covdd = np.linalg.cond(covdd)

    covddinv = compute_inverse(covdd)

    llist = []

    for absd in absdsamples:
        llist.append(choLL(absd, covdd, lam=lam))

    llist = np.array(llist)

    outputpath = os.path.join(outputfolder,
                              f"covdd_{elemid}_Ns{nelemsamples}.json")
    covddict = {
            "elemid": elem.id,
            "kp1": elem.kp1,
            "ph1": elem.ph1,
            "covdd": covdd.tolist(),
            "dnorm": elem.dnorm,
            "evals_covdd": evals_covdd.tolist(),
            "evecs_covdd": evecs_covdd.tolist(),
            "cond_covdd": cond_covdd,
            "covddinv": covddinv.tolist(),
            "llist": llist.tolist(),
            }

    with open(outputpath, 'w') as json_file:
        json.dump(covddict, json_file)

    return covddict


def test_run_kifit(elemid="Camin_testdata",
                   Nsamples=100,
                   x=0,
                   detstr='gkp',
                   dim=3,
                   read=False):

    configpath = os.path.join(inputfolder,
                              f"{elemid}_Ns{Nsamples}_x{x}_config.json")

    if os.path.exists(configpath):

        from kifit.config import RunParams
        from kifit.run import Runner

        params = RunParams(configuration_file=configpath)
        runner = Runner(params)

        if (not read
            or not os.path.exists(runner.config.paths.fit_output_path(x))):
            runner.run()

        fit_output = runner.config.paths.read_fit_output(x)
        det_output = runner.config.paths.read_det_output(detstr, dim, x)

        return fit_output, det_output

    else:
        raise ImportError(f"Please provide configuration file {configpath}.")


def kifitswap(
        elem,
        inputparamsamples,
        fitparamsamples,
        alphasamples,
        min_percentile,
        elem_swap=None,
        only_inputparams=False,
        # only_fitparams=False,
        lam=0,
        symm=False):

    elem_ll_elemfit = []
    elem_swap_ll_elemfit = []

    for alphaNP in alphasamples:

        elem_absdsamples = []

        if elem_swap is not None:
            elem_swap_absdsamples = []

        for i, fitparams in enumerate(fitparamsamples):

            # update fitparams
            kp1 = fitparams[0]
            ph1 = fitparams[1]

            if np.isclose(ph1, 0., atol=0, rtol=1e-17):
                raise Exception("ph1 is too close to zero")

            kp1swap = kp1
            ph1swap = np.arctan(1 / np.tan(ph1))

            if not only_inputparams:
                elem._update_fit_params([kp1, ph1, alphaNP])
                if elem_swap is not None:
                    elem_swap._update_fit_params([kp1swap, ph1swap, alphaNP])
            else:
                elem.alphaNP = alphaNP
                if elem_swap is not None:
                    elem_swap.alphaNP = alphaNP

            # update inputparams
            # if not only_fitparams:
            inputparams = inputparamsamples[i]
            elem._update_elem_params(inputparams)
            elem_absdsamples.append(elem.absd(symm))

            if elem_swap is not None:
                # if not only_fitparams:
                inputparamswap = swap_inputparams(inputparams,
                                                  elem.nisotopepairs,
                                                  elem.ntransitions)
                elem_swap._update_elem_params(inputparamswap)
                elem_swap_absdsamples.append(elem_swap.absd(symm))

        elem_ll = get_llist_elemsamples(elem_absdsamples, lam=lam)
        elem_ll_elemfit.append(np.percentile(elem_ll, min_percentile))

        elem_covmat = (np.cov(np.array(elem_absdsamples), rowvar=False)
                           + lam * np.eye(elem.nisotopepairs))

        if elem_swap is not None:
            assert np.allclose(np.mean(elem_absdsamples),
                               np.mean(elem_swap_absdsamples),
                               atol=0, rtol=.2)
            assert np.allclose(np.percentile(elem_absdsamples, 5),
                               np.percentile(elem_swap_absdsamples, 5),
                               atol=0, rtol=2)
            elem_swap_covmat = (np.cov(np.array(elem_swap_absdsamples),
                                rowvar=False)
                                + lam * np.eye(elem.nisotopepairs))

            assert (compute_spectral_difference(elem_covmat,
                                                elem_swap_covmat) < .5)

            assert (compute_Kullback_Leibler_divergence(elem_covmat,
                                                        elem_swap_covmat) < .2)

            elem_swap_ll = get_llist_elemsamples(elem_swap_absdsamples, lam=lam)

            assert np.allclose(elem_ll, elem_swap_ll, atol=0, rtol=1)

            elem_swap_ll_elemfit.append(np.percentile(elem_swap_ll,
                                                      min_percentile))

    elem_delchisq_elemfit = get_delchisq(elem_ll_elemfit, minll=None)

    if elem_swap is None:
        return elem_delchisq_elemfit

    else:
        elem_swap_delchisq_elemfit = get_delchisq(elem_swap_ll_elemfit,
                                                  minll=None)
        return elem_delchisq_elemfit, elem_swap_delchisq_elemfit

def swap_varying_elemparams(only_inputparams=False, symm=False):

    # varying both input parameters and fit parameters

    camin = Elem('Camin')
    camin_swap = Elem('Camin_swap')

    nalphasamples = 150
    nelemsamples = 100
    min_percentile = 5
    delchisqcrit = get_delchisq_crit(2)

    inputparamsamples, fitparamsamples = generate_elemsamples(camin, nelemsamples)
    inputparamsamples_swap, fitparamsamples_swap = generate_elemsamples(
            camin, nelemsamples)

    if only_inputparams:
        camin.set_alphaNP_init(0, 5e-11)

    else:
        camin.set_alphaNP_init(0, 5e-11)

    alphasamples = generate_alphaNP_samples(
        camin,
        nalphasamples,
        search_mode="normal")

    camin_delchisq_caminfit, camin_swap_delchisq_caminfit = kifitswap(
        camin,
        inputparamsamples,
        fitparamsamples,
        alphasamples,
        min_percentile,
        elem_swap=camin_swap,
        only_inputparams=only_inputparams,
        symm=symm)
    camin_confint = get_confint(alphasamples, camin_delchisq_caminfit, delchisqcrit)

    camin_swap_delchisq_caminswapfit, camin_delchisq_caminswapfit = kifitswap(
        camin,
        inputparamsamples_swap,
        fitparamsamples_swap,
        alphasamples,
        min_percentile,
        elem_swap=camin_swap,
        only_inputparams=only_inputparams,
        symm=symm)
    camin_swap_confint = get_confint(alphasamples,
                                     camin_swap_delchisq_caminswapfit,
                                     delchisqcrit)

    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredText

    fig, ax = plt.subplots()

    ax.scatter(alphasamples, camin_delchisq_caminfit, color='r',
               label="Ca 21, Ca 21 samples", s=15)
    ax.scatter(alphasamples, camin_swap_delchisq_caminfit, color='b',
               label="Ca 12, Ca 21 samples", s=5)
    ax.scatter(alphasamples, camin_delchisq_caminswapfit, color='orange',
               label="Ca 21, Ca 12 samples", s=15)
    ax.scatter(alphasamples, camin_swap_delchisq_caminswapfit, color='c',
               label="Ca 12, Ca 12 samples", s=5)

    # fit 2-sigma region
    ax.axhline(y=0, color="k", lw=1, ls="-")
    ax.axhline(y=delchisqcrit, color="r", lw=1, ls="--")
    ax.axvline(x=camin_confint[0], color="b", lw=1, ls="--")
    ax.axvline(x=camin_confint[1], color="b", lw=1, ls="--")
    ax.axvline(x=camin_swap_confint[0], color="green", lw=1, ls="--")
    ax.axvline(x=camin_swap_confint[1], color="green", lw=1, ls="--")

    ax.set_xlabel(r"$\alpha_{\mathrm{NP}} / \alpha_{\mathrm{EM}}$",
                  fontsize=axislabelsize)
    ax.set_ylabel(r"$\Delta \chi^2$", fontsize=axislabelsize)

    camin_alpha_det, camin_LB_det, camin_UB_det = get_det_vals(
            camin, nelemsamples, 3, "gkp")
    camin_swap_alpha_det, camin_swap_LB_det, camin_swap_UB_det = get_det_vals(
            camin, nelemsamples, 3, "gkp")


    detext = (r"$\bf{dim~3~GKP:}$" + "\n "
              + "Ca 21:\n"
              + r"$\langle\frac{\alpha_{\mathrm{NP}}}{\alpha_{\mathrm{EM}}}\rangle =$"
              + f"{camin_alpha_det[0]:.1e}\n"
              + r"$\frac{\alpha_{\mathrm{NP}}}{\alpha_{\mathrm{EM}}}\in$ ["
              + f"{camin_LB_det[0]:.1e}, {camin_UB_det[0]:.1e}"
              + "]\n"
              + "Ca 12:\n"
              + r"$\langle\frac{\alpha_{\mathrm{NP}}}{\alpha_{\mathrm{EM}}}\rangle =$"
              + f"{camin_alpha_det[0]:.1e}\n"
              + r"$\frac{\alpha_{\mathrm{NP}}}{\alpha_{\mathrm{EM}}}\in$ ["
              + f"{camin_swap_LB_det[0]:.1e}, {camin_swap_UB_det[0]:.1e}"
              + "]")

    light_grey = "#D7D7D7"
    textbox_props = dict(boxstyle='round', facecolor="white", edgecolor=light_grey)
    anchored_text = AnchoredText(detext, loc="upper right", frameon=False,
                                 prop=dict(bbox=textbox_props))
    # ax.add_artist(anchored_text)
    # plt.legend(loc="upper left")
    plt.legend(loc="upper center", fontsize=fsize)
    ax.tick_params(axis="both", which="major", labelsize=fsize)
    ax.xaxis.get_offset_text().set_fontsize(fsize)

    varstr = "inputparamvar" if only_inputparams else "elemparamvar"
    symmstr = "symm_" if symm else ""

    plotpath = os.path.join(
        outputfolder,
        f"mc_output_Camin_01_10_{symmstr}x{camin.x}_{varstr}.pdf")
    plt.savefig(plotpath)

    # if not only_inputparams:
    ax.set_ylim(0, 20)
    ax.set_xlim(-1e-10, 1e-10)

    plotpath = os.path.join(
        outputfolder,
        f"mc_output_Camin_10_01_{symmstr}x{camin.x}_{varstr}_zoom.pdf")
    plt.savefig(plotpath)


def test_swap():
    swap_varying_elemparams(only_inputparams=False, symm=False)
    swap_varying_elemparams(only_inputparams=True, symm=False)
    swap_varying_elemparams(only_inputparams=False, symm=True)
    swap_varying_elemparams(only_inputparams=True, symm=True)


def test_Ca24min_mod():

    # varying both input parameters and fit parameters

    camin = Elem('Ca24min_mod')

    nalphasamples = 150
    nelemsamples = 100
    min_percentile = 0
    delchisqcrit = get_delchisq_crit(3)

    inputparamsamples, fitparamsamples = generate_elemsamples(camin, nelemsamples)

    camin.set_alphaNP_init(0, 5e-11)

    alphasamples = generate_alphaNP_samples(
        camin,
        nalphasamples,
        search_mode="normal")

    delchisqs = kifitswap(
        camin,
        inputparamsamples,
        fitparamsamples,
        alphasamples,
        min_percentile)

    confint = get_confint(alphasamples, delchisqs, delchisqcrit)

    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredText

    fig, ax = plt.subplots()

    ax.scatter(alphasamples, delchisqs, s=5)

    # fit 2-sigma region
    ax.axhline(y=0, color="k", lw=1, ls="-")
    ax.axhline(y=delchisqcrit, color="r", lw=1, ls="--")
    ax.axvline(x=confint[0], color="b", lw=1, ls="--")
    ax.axvline(x=confint[1], color="b", lw=1, ls="--")

    ax.set_xlabel(r"$\alpha_{\mathrm{NP}} / \alpha_{\mathrm{EM}}$",
                  fontsize=axislabelsize)
    ax.set_ylabel(r"$\Delta \chi^2$", fontsize=axislabelsize)

    camin_alpha_det, camin_LB_det, camin_UB_det = get_det_vals(
            camin, nelemsamples, 3, "gkp")


    detext = (r"$\bf{kifit:}$" + "\n"
              + r"$\frac{\alpha_{\mathrm{NP}}}{\alpha_{\mathrm{EM}}}\in$ ["
              + f"{confint[0]:.1e}, {confint[1]:.1e}"
              + "] \n \n"
              + r"$\bf{dim~3~GKP:}$" + "\n "
              + r"$\langle\frac{\alpha_{\mathrm{NP}}}{\alpha_{\mathrm{EM}}}\rangle =$"
              + f"{camin_alpha_det[0]:.1e}\n"
              + r"$\frac{\alpha_{\mathrm{NP}}}{\alpha_{\mathrm{EM}}}\in$ ["
              + f"{camin_LB_det[0]:.1e}, {camin_UB_det[0]:.1e}"
              + "]")


    light_grey = "#D7D7D7"
    textbox_props = dict(boxstyle='round', facecolor="white", edgecolor=light_grey)
    anchored_text = AnchoredText(detext, loc="upper right", frameon=False,
                                 prop=dict(bbox=textbox_props))
    ax.add_artist(anchored_text)
    # plt.legend(loc="upper center", fontsize=fsize)
    ax.tick_params(axis="both", which="major", labelsize=fsize)
    ax.xaxis.get_offset_text().set_fontsize(fsize)

    # varstr = "inputparamvar" if only_inputparams else "elemparamvar"
    # symmstr = "symm_" if symm else ""
    # ax.set_ylim(0, 1e5)
    # ax.set_xlim(-1e-8, 1e-8)

    plotpath = os.path.join(
        outputfolder,
        f"mc_output_Ca24min_mod_x{camin.x}.pdf")
    plt.savefig(plotpath)

    # if not only_inputparams:
    ax.set_ylim(0, 20)
    ax.set_xlim(-1e-10, 1e-10)

    plotpath = os.path.join(
        outputfolder,
        f"mc_output_Ca24min_mod_x{camin.x}_zoom.pdf")
    plt.savefig(plotpath)


def test_lam():

    lamvals = [1e-50, 1e-17, 1e-11]

    nalphasamples = 100
    nelemsamples = 100
    min_percentile = 0

    camin = Elem('Camin')

    inputparamsamples, fitparamsamples = generate_elemsamples(camin, nelemsamples)

    # camin.set_alphaNP_init(1.5e-9, 1e-9)
    # camin.set_alphaNP_init(5.898063988306249e-12, 1.656617885637278e-09)
    camin.set_alphaNP_init(0, 5e-11)


    alphasamples = generate_alphaNP_samples(
        camin,
        nalphasamples,
        search_mode="normal")

    camin_delchisq_symm = []
    camin_delchisq = []

    for lamval in lamvals:

        camin_ll_lam_symm = []
        camin_ll_lam = []

        for alphaNP in alphasamples:

            camin_absdsamples_symm = []
            camin_absdsamples = []

            for i, fitparams in enumerate(fitparamsamples):

                fitparams = fitparams.tolist()
                fitparams.append(alphaNP)

                if np.isclose(fitparams[1], 0., atol=0, rtol=1e-17):
                    raise Exception("ph1 is too close to zero")

                camin._update_fit_params(fitparams)
                camin._update_elem_params(inputparamsamples[i])

                camin_absdsamples_symm.append(camin.absd(True))
                camin_absdsamples.append(camin.absd(False))

            camin_ll_symm = get_llist_elemsamples(camin_absdsamples_symm, lam=lamval)
            camin_ll = get_llist_elemsamples(camin_absdsamples, lam=lamval)

            camin_ll_lam_symm.append(np.percentile(camin_ll_symm, min_percentile))
            camin_ll_lam.append(np.percentile(camin_ll, min_percentile))

        camin_delchisq_symm.append(get_delchisq(camin_ll_lam_symm, minll=None))
        camin_delchisq.append(get_delchisq(camin_ll_lam, minll=None))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    for l, lam in enumerate(lamvals):
        ax.scatter(alphasamples, camin_delchisq_symm[l], label=f"lam={lam} symm",
                   s=5, marker="v")
        ax.scatter(alphasamples, camin_delchisq[l], label=f"lam={lam}", s=5,
                   marker="o")

    ax.set_xlabel(r"$\alpha_{\mathrm{NP}} / \alpha_{\mathrm{EM}}$",
                  fontsize=axislabelsize)
    ax.set_ylabel(r"$\Delta \chi^2$", fontsize=axislabelsize)

    # plt.legend(fontsize=fsize)

    ax.set_ylim(0, 1)
    ax.set_xlim(-.5e-10, .5e-10)

    ax.tick_params(axis="both", which="major", labelsize=fsize)
    ax.xaxis.get_offset_text().set_fontsize(fsize)

    plotpath = os.path.join(outputfolder,
                            f"mc_output_Camin_lam_x{camin.x}.pdf")
    plt.savefig(plotpath, dpi=1000)

    ax.set_ylim(0, 1)
    ax.set_xlim(-.5e-10, .5e-10)
    ax.tick_params(axis="both", which="major", labelsize=fsize)
    ax.xaxis.get_offset_text().set_fontsize(fsize)

    plotpath = os.path.join(outputfolder,
                            f"mc_output_Camin_lam_x{camin.x}_zoom.pdf")
    plt.savefig(plotpath)



def plot_elemvar_vs_elemfitvar(symm=False):
    nalphasamples = 100  # 500
    nelemsamples = 100  # 500
    min_percentile = 0
    delchisqcrit = get_delchisq_crit(2)

    camin = Elem('Camin')

    alpha_det, LB_det, UB_det = get_det_vals(camin, nelemsamples, 3, "gkp")

    camin.set_alphaNP_init(0, 1e-10)

    alphasamples = generate_alphaNP_samples(
            camin,
            nalphasamples,
            search_mode="normal")

    inputparamsamples, fitparamsamples = generate_elemsamples(camin, nelemsamples)

    delchisqs_elemvar = kifitswap(camin,
                                  inputparamsamples,
                                  fitparamsamples,
                                  alphasamples,
                                  min_percentile=min_percentile,
                                  only_inputparams=True,
                                  symm=symm)
    confint_elemvar = get_confint(alphasamples, delchisqs_elemvar, delchisqcrit)

    delchisqs_elemfitvar = kifitswap(camin,
                                     inputparamsamples,
                                     fitparamsamples,
                                     alphasamples,
                                     min_percentile=min_percentile,
                                     only_inputparams=False,
                                     symm=symm)
    confint_elemfitvar = get_confint(alphasamples, delchisqs_elemfitvar, delchisqcrit)
    #
    # delchisqs_fitvar = kifitswap(camin,
    #                              inputparamsamples,
    #                              fitparamsamples,
    #                              alphasamples,
    #                              min_percentile=min_percentile,
    #                              only_fitparams=True,
    #                              lam=1e-17,
    #                              symm=symm)
    # confint_fitvar = get_confint(alphasamples, delchisqs_fitvar, delchisqcrit)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(alphasamples, delchisqs_elemvar,
               label=(r"varying input parameters only: $\alpha_{\mathrm{NP}}\in$"
                      + f"[{confint_elemvar[0]:.1e},{confint_elemvar[1]:.1e}]"),
               s=5, color='b')
    ax.scatter(alphasamples, delchisqs_elemfitvar,
               label=(r"varying input params. & $K^\perp, \phi$: "
                      + r"$\alpha_{\mathrm{NP}}\in$"
                      +
                      f"[{confint_elemfitvar[0]:.1e},{confint_elemfitvar[1]:.1e}]"),
               s=5, color='purple')
    # ax.scatter(alphasamples, delchisqs_fitvar,
    #            label=(r"varying fit params. $K^\perp, \phi$ only:"
    #                   + r"$\alpha_{\mathrm{NP}}\in$"
    #                   + f"[{confint_fitvar[0]:.1e},{confint_fitvar[1]:.1e}]"),
    #            s=5, color='orange')

    # fit 2-sigma region
    ax.axhline(y=0, color="k", lw=1, ls="-")
    ax.axhline(y=delchisqcrit, color="r", lw=1, ls="--")
    ax.axvline(x=confint_elemvar[0], color="b", lw=1, ls="--")
    ax.axvline(x=confint_elemvar[1], color="b", lw=1, ls="--")
    ax.axvline(x=confint_elemfitvar[0], color="orange", lw=1, ls="--")
    ax.axvline(x=confint_elemfitvar[1], color="orange", lw=1, ls="--")


    # determinant 2-sigma region
    # ax.axvline(x=alpha_det[0], ls="--", color='purple',
    #            label=("dim-3 GKP: "
    #                   + r"$\langle\alpha_{\mathrm{NP}}\rangle = $ "
    #         + (f"{alpha_det[0]:.1e}" if not np.isnan(alpha_det[0])
    #            else "-")))
    # ax.axvline(x=LB_det[0], ls="--", color='b',
    #            label=("dim-3 GKP: "
    #                   + r"$\alpha_{\mathrm{NP}}\in$ ["
    #         + (f"{LB_det[0]:.1e}" if not np.isnan(LB_det[0]) else "-")
    #         + ", "
    #         + (f"{UB_det[0]:.1e}" if not np.isnan(UB_det[0]) else "-")
    #         + "]"))
    # ax.axvline(x=UB_det, ls="--", color='b')

    plt.title("dim-3 GKP: "
                      + r"$\alpha_{\mathrm{NP}}\in$ ["
            + (f"{LB_det[0]:.1e}" if not np.isnan(LB_det[0]) else "-")
            + ", "
            + (f"{UB_det[0]:.1e}" if not np.isnan(UB_det[0]) else "-")
            + "]", fontsize=fsize)

    # admin
    ax.set_xlabel(r"$\alpha_{\mathrm{NP}} / \alpha_{\mathrm{EM}}$",
                  fontsize=axislabelsize)
    ax.set_ylabel(r"$\Delta \chi^2$", fontsize=axislabelsize)
    ax.set_xlim(2 * confint_elemfitvar[0], 2 * confint_elemfitvar[1])
    ax.set_ylim(0, 10)
    ax.tick_params(axis="both", which="major", labelsize=fsize)
    ax.xaxis.get_offset_text().set_fontsize(fsize)
    plt.legend(loc='upper center', fontsize=fsize)
    plotpath = os.path.join(outputfolder,
                            f"mc_output_elemvar_vs_elemfitvar_x{camin.x}"
                            + ("_symm" if symm else "")
                            + ".pdf")
    plt.savefig(plotpath, dpi=1000)


def test_elemvar_vs_elemfitvar_symm_vs_asymm():
    plot_elemvar_vs_elemfitvar(symm=True)
    plot_elemvar_vs_elemfitvar(symm=False)


if __name__ == "__main__":
    test_sample_woNP()
    test_sample_wNP()
    test_covmats()
    test_sampling()
    test_d_swap_varying_inputparams()
    test_alphaNP_dependence_covdd()
    test_write_covdd()
    test_run_kifit()
    test_swap()
    test_Ca24min_mod()
    test_lam()
    test_elemvar_vs_elemfitvar_symm_vs_asymm()
