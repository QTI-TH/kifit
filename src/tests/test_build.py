import numpy as np
import os
from math import factorial
from pprint import pprint
from kifit.build import Elem, ElemCollection, Levi_Civita_generator, LeviCivita
from kifit.build import perform_odr, perform_linreg, det_64
from kifit.fitools import (
    generate_elemsamples, generate_alphaNP_samples,
    get_llist_elemsamples, get_delchisq)
from Mathematica_crosschecks import *
from itertools import permutations, combinations, product


def test_load_all():
    all_element_data = Elem.load_all()

    assert np.allclose(
        np.nan_to_num(all_element_data['Ca_testdata'].nu).sum(axis=1),
        [8.44084572e+08, 1.68474592e+09, 3.38525526e+09],
        rtol=1e-8,
    )


def test_load_individual():
    ca = Elem('Ca_testdata')

    # overwrite nuclear masses with atomic masses
    ca.m_a_in = ca.isotope_data[1]
    ca.sig_m_a_in = ca.isotope_data[2]
    ca.m_ap_in = ca.isotope_data[4]
    ca.sig_m_ap_in = ca.isotope_data[5]

    pprint(ca)

    Ca = Elem('Ca_testdata')

    assert np.all(np.nan_to_num(ca.nu) == np.nan_to_num(Ca.nu))
    assert np.isclose(np.sum(ca.m_a), 119.88777255, atol=0, rtol=1e-5)
    assert np.isclose(np.sum(ca.m_ap), 133.86662192999998, atol=0, rtol=1e-5)
    assert np.isclose(np.sum(ca.sig_m_a_in**2), 1.452e-15, atol=0, rtol=1e-5)
    assert np.isclose(np.sum(ca.sig_m_ap_in**2), 1.15961e-13, atol=0, rtol=1e-5)

    assert ca.nisotopepairs == len(ca.ap_nisotope)
    assert ca.nisotopepairs == len(ca.m_a)
    assert ca.nisotopepairs == len(ca.m_ap)
    assert ca.nisotopepairs == len(ca.nu)
    assert ca.nisotopepairs == len(ca.sig_nu)
    assert ca.ntransitions == ca.sig_nu.shape[1]

    assert len(ca.m_a) == len(ca.m_ap)
    assert all(u != v for u, v in zip(ca.m_a, ca.m_ap))

    assert (len(ca.Xcoeff_data) > 0), len(ca.Xcoeff_data)
    assert (len(ca.sig_Xcoeff_data) > 0), len(ca.sig_Xcoeff_data)

    assert np.all(ca.Xcoeff_data == ca.sig_Xcoeff_data)  # adjust in future

    assert len(ca.Xvec) == ca.ntransitions, len(ca.Xvec)
    assert len(ca.sig_Xvec) == ca.ntransitions, len(ca.sig_Xvec)

    for x in range(len(ca.Xcoeff_data)):
        assert len(ca.Xcoeff_data[x]) == ca.ntransitions + 1, len(ca.Xcoeff_data[x])
        assert len(ca.sig_Xcoeff_data[x]) == ca.ntransitions + 1

    assert (ca.nu.size == ca.nisotopepairs * ca.ntransitions)
    assert np.allclose(ca.nutil_in, mnu_Mathematica, atol=0, rtol=1e-14)
    assert np.allclose(ca.sig_nutil_in, sig_mnu_Mathematica, atol=0, rtol=1e-14)
    # assert (np.sum(ca.corr_nu_nu) == ca.nisotopepairs * ca.ntransitions)
    # assert (np.sum(ca.corr_m_m) == 1)
    # assert (np.trace(ca.corr_m_mp) == 0)
    # assert (np.trace(ca.corr_mp_mp) == ca.nisotopepairs)
    # assert (np.trace(ca.corr_X_X) == ca.ntransitions)
    assert (ca.muvec[0] == 1 / ca.m_a[0] - 1 / ca.m_ap[0])
    assert (len(ca.muvec) == ca.nisotopepairs)
    assert (ca.gammatilvec.size == ca.nisotopepairs)
    assert np.allclose(ca.gammatilvec, np.array([(ca.a_nisotope[a] - ca.ap_nisotope[a])
        / ca.muvec[a] for a in ca.range_a]), atol=0, rtol=1e-5)
    assert (ca.diff_np_term(False).size == ca.nisotopepairs * ca.ntransitions)
    assert (ca.diff_np_term(True).size == ca.nisotopepairs * ca.ntransitions)
    assert np.all([i.is_integer() for i in ca.a_nisotope])
    assert np.all([i.is_integer() for i in ca.ap_nisotope])
    assert (ca.F1.size == ca.ntransitions)
    assert (ca.F1sq == ca.F1 @ ca.F1)
    assert np.isclose(ca.F1[0], 1, atol=0, rtol=1e-17)
    assert (ca.secph1.size == ca.ntransitions)
    assert np.isclose(ca.secph1[0], 0, atol=0, rtol=1e-17)
    assert (ca.Kperp1.size == ca.ntransitions)
    assert np.isclose(ca.Kperp1[0], 0, atol=0, rtol=1e-17)
    assert (ca.X1.size == ca.ntransitions)
    assert np.isclose(ca.X1[0], 0, atol=0, rtol=1e-17)
    assert (sum(ca.range_i) == sum(ca.range_j))
    assert (len(ca.range_i) == len(ca.range_j) + 1)


def test_linfit():

    ca = Elem('Ca_testdata')

    (betas_odr, sig_betas_odr, kperp1s_odr, ph1s_odr,
        sig_kperp1s_odr, sig_ph1s_odr, cov_kperp1_ph1s) = perform_odr(
        ca.nutil_in, ca.sig_nutil_in)

    (betas_linreg, sig_betas_linreg, kperp1s_linreg, ph1s_linreg,
        sig_kperp1s_linreg, sig_ph1s_linreg) = perform_linreg(ca.nutil_in)

    assert betas_odr.shape == (ca.ntransitions - 1, 2)
    assert betas_linreg.shape == (ca.ntransitions - 1, 2)

    assert np.all(np.isclose(betas_odr, betas_linreg, atol=0, rtol=1e-2))
    assert np.all(np.isclose(sig_betas_odr, sig_betas_linreg, atol=0, rtol=1))
    assert np.all(np.isclose(kperp1s_odr, kperp1s_linreg, atol=0, rtol=1e-2))
    assert np.all(np.isclose(ph1s_odr, ph1s_linreg, atol=0, rtol=1e-2))
    assert np.all(np.isclose(sig_kperp1s_odr, sig_kperp1s_linreg, atol=0, rtol=1))
    assert np.all(np.isclose(sig_ph1s_odr, sig_ph1s_linreg, atol=0, rtol=1))

    xvals = (ca.nutil_in.T[0]).astype(np.float64)
    yvals = (ca.nutil_in[:, 1:]).astype(np.float64)

    betas_dat = np.array([np.polyfit(xvals, yvals[:, i], 1) for i in
        range(yvals.shape[1])])

    assert betas_dat.shape == (ca.ntransitions -1, 2)
    assert np.all(np.isclose(betas_dat, betas_odr, atol=0, rtol=1e-2))


def test_set_fit_params():
    ca = Elem('Ca_testdata')

    kappaperp1temp = np.array(list(range(ca.ntransitions - 1)),
                              dtype=np.float128)
    ph1temp = np.array(
            [np.pi / 2 - np.pi / (i + 2) for i in range(ca.ntransitions - 1)],
            dtype=np.float128)
    alphaNPtemp = 1 / (4 * np.pi)
    thetatemp = np.concatenate((kappaperp1temp, ph1temp, alphaNPtemp),
        axis=None)
    ca._update_fit_params(thetatemp)
    assert (np.sum(ca.kp1) == np.sum(kappaperp1temp))
    assert (np.sum(ca.Kperp1) == np.sum(kappaperp1temp))
    assert (len(ca.Kperp1) == (len(kappaperp1temp) + 1))
    assert (np.sum(ca.ph1) == np.sum(ph1temp))
    assert (len(ca.ph1) == len(ph1temp))
    assert (ca.alphaNP == alphaNPtemp)

    assert all(ca.F1[1:] == np.tan(ph1temp))

    theta_Mathematica = np.concatenate((kappaperp1nit_Mathematica,
        ph1nit_Mathematica, alphaNP_Mathematica), axis=None)
    ca._update_fit_params(theta_Mathematica)
    assert (np.sum(ca.Kperp1) == np.sum(np.array(kappaperp1nit_Mathematica,
                                                 dtype=np.float128)))
    assert (np.sum(ca.kp1)
            == np.sum(np.array(kappaperp1nit_Mathematica, dtype=np.float128)))
    assert (np.sum(ca.ph1) == np.sum(np.array(ph1nit_Mathematica,
                                              dtype=np.float128)))
    assert (ca.alphaNP == alphaNP_Mathematica)
    assert (ca.F1[1:] == np.tan(np.array(ph1nit_Mathematica,
                                         dtype=np.float128))).all()
    assert np.isclose(ca.F1sq, F1_Mathematica @ F1_Mathematica, atol=0, rtol=1e-15)

    theta_LL_Mathematica = np.concatenate((kappaperp1nit_LL_Mathematica,
            ph1nit_LL_Mathematica, 0.), axis=None)
    ca._update_fit_params(theta_LL_Mathematica)
    assert (np.sum(ca.Kperp1) == np.sum(np.array(kappaperp1nit_LL_Mathematica,
                                                 dtype=np.float128)))
    assert (np.sum(ca.ph1) == np.sum(np.array(ph1nit_LL_Mathematica,
                                              dtype=np.float128)))
    assert (ca.alphaNP == alphaNP_Mathematica)
    assert (ca.F1[1:] == np.tan(np.array(ph1nit_LL_Mathematica,
                                         dtype=np.float128))).all()
    assert np.isclose(ca.F1sq, F1_LL_Mathematica @ F1_LL_Mathematica,
        atol=0, rtol=1e-15)


def test_constr_dvec():
    ca = Elem('Ca_testdata')

    # overwrite nuclear masses with atomic masses
    ca.m_a_in = ca.isotope_data[1]
    ca.sig_m_a_in = ca.isotope_data[2]
    ca.m_ap_in = ca.isotope_data[4]
    ca.sig_m_ap_in = ca.isotope_data[5]


    theta_LL_Mathematica = np.concatenate((kappaperp1nit_LL_Mathematica,
            ph1nit_LL_Mathematica, 0.), axis=None)

    # without NP
    ca._update_fit_params(theta_LL_Mathematica)
    assert (np.isclose(spp, spm, atol=0, rtol=1e-7) for (spp, spm) in zip(ca.secph1,
        secph1nit_LL_Mathematica)), (ca.secph1, secph1nit_LL_Mathematica)
    assert np.allclose(ca.muvec, muvec_Mathematica, atol=0, rtol=1e-10)

    D_a1i_python = [[ca.D_a1i(a, i, True) for i in ca.range_i] for a in ca.range_a]

    assert np.allclose(D_a1i_Mathematica, D_a1i_python, atol=0, rtol=1e-14)
    assert np.allclose(np.array(dmat_Mathematica) / ca.dnorm, ca.dmat(True),
        atol=0, rtol=1e-8)

    # with NP
    theta_LL_Mathematica_1 = np.concatenate((kappaperp1nit_LL_Mathematica,
            ph1nit_LL_Mathematica, 1.), axis=None)

    ca._update_fit_params(theta_LL_Mathematica_1)

    # avg_NP_term_alphaNP_1_Mathematica = np.average(
        # NP_term_alphaNP_1_Mathematica, axis=1)

    assert np.allclose(ca.diff_np_term(True), diff_NP_term_alphaNP_1_Mathematica,
        atol=0, rtol=1e-7)

    D_a1i_alphaNP_1_python = [[ca.D_a1i(a, i, True) for i in ca.range_i] for a in ca.range_a]

    assert np.allclose(D_a1i_alphaNP_1_Mathematica, D_a1i_alphaNP_1_python,
        atol=0, rtol=1e-13)
    assert np.allclose(np.array(dmat_alphaNP_1_Mathematica) / ca.dnorm,
                       ca.dmat(True), atol=0, rtol=1e-13)

    absd_explicit = np.array([np.sqrt(np.sum(
        np.fromiter([ca.d_ai(a, i, True)**2
                     for i in ca.range_i], float)))
                              for a in ca.range_a]) / ca.dnorm
    assert np.allclose(ca.absd(True), absd_explicit, atol=0, rtol=1e-15)


def test_swap_elem():

    camin_ref1 = Elem('Camin', reference_transition_index=1)
    camin_swap = Elem('Camin_swap')

    assert np.allclose(camin_ref1.nu, camin_swap.nu, atol=0,
                       rtol=1e-20)
    assert np.allclose(camin_ref1.sig_nu, camin_swap.sig_nu, atol=0,
                       rtol=1e-20)

    assert np.allclose(camin_ref1.Xvec, camin_swap.Xvec, atol=0,
                       rtol=1e-20)
    assert np.allclose(camin_ref1.sig_Xvec, camin_swap.sig_Xvec, atol=0,
                       rtol=1e-20)
    assert np.allclose(camin_ref1.mphi, camin_swap.mphi, atol=0,
                       rtol=1e-20)

    assert np.allclose(camin_ref1.m_a_in, camin_swap.m_a_in, atol=0,
                       rtol=1e-20)
    assert np.allclose(camin_ref1.sig_m_a_in, camin_swap.sig_m_a_in, atol=0,
                       rtol=1e-20)
    assert np.allclose(camin_ref1.m_ap_in, camin_swap.m_ap_in, atol=0,
                       rtol=1e-20)
    assert np.allclose(camin_ref1.sig_m_ap_in, camin_swap.sig_m_ap_in, atol=0,
                       rtol=1e-20)

def levi_civita_tensor(d):
    arr = np.zeros([d for _ in range(d)])
    for x in permutations(tuple(range(d))):
        mat = np.zeros((d, d), dtype=np.int32)
        for i, j in enumerate(x):
            mat[i][j] = 1
        arr[x] = int(np.linalg.det(mat))
    return arr


def test_levi_civita():
    eps2_gen = Levi_Civita_generator(2)
    eps2 = levi_civita_tensor(2)

    eps2_gentens = np.zeros(eps2.shape)
    for inds, sign in eps2_gen:
        eps2_gentens[inds] = sign
    assert np.array_equal(eps2, eps2_gentens)

    eps4_gen = Levi_Civita_generator(4)
    eps4 = levi_civita_tensor(4)

    eps4_gentens = np.zeros(eps4.shape)
    for inds, sign in eps4_gen:
        eps4_gentens[inds] = sign
    assert np.array_equal(eps4, eps4_gentens)


def check_alphaNP_GKP(elem, dim):

    alphalist = []

    for a_inds, i_inds in product(combinations(elem.range_a, dim),
            combinations(elem.range_i, dim - 1)):

        numat = elem.nutil[np.ix_(a_inds, i_inds)]
        mumat = elem.mutilvec[np.ix_(a_inds)]
        Xmat = elem.Xvec[np.ix_(i_inds)]
        hmat = elem.gammatilvec[np.ix_(a_inds)]

        vol_data = det_64(np.c_[numat, mumat])

        vol_alphaNP1 = 0
        for i, eps_i in LeviCivita(dim - 1):
            vol_alphaNP1 += (eps_i * det_64(np.c_[
                Xmat[i[0]] * hmat,
                np.array([numat[:, i[s]] for s in range(1, dim - 1)]).T,
                mumat]))
        alphalist.append(vol_data / vol_alphaNP1)

    alphalist = factorial(dim - 2) * np.array(alphalist)

    return alphalist


def check_alphaNP_NMGKP(elem, dim):

    alphalist = []

    for a_inds, i_inds in product(combinations(elem.range_a, dim),
            combinations(elem.range_i, dim)):

        numat = elem.nutil[np.ix_(a_inds, i_inds)]
        Xmat = elem.Xvec[np.ix_(i_inds)]
        hmat = elem.gammatilvec[np.ix_(a_inds)]

        vol_data = det_64(numat)

        vol_alphaNP1 = 0
        for i, eps_i in LeviCivita(dim):
            vol_alphaNP1 += (eps_i * det_64(np.c_[
                Xmat[i[0]] * hmat,
                np.array([numat[:, i[s]] for s in range(1, dim)]).T]))
        alphalist.append(vol_data / vol_alphaNP1)

    alphalist = factorial(dim - 1) * np.array(alphalist)

    return alphalist


def test_alphaNP_GKP():
    ca = Elem('Ca_testdata')
    assert np.isclose(ca.alphaNP_GKP(), check_alphaNP_GKP(ca, 3)[0], atol=0,
        rtol=1e-21)

    yb1 = Elem("strongest_Yb_Kyoto_MIT_GSI_2022")
    yb1_alphaNP = 5.068061e-11
    assert np.isclose(yb1.alphaNP_GKP(), yb1_alphaNP, atol=0, rtol=1e-5)

    yb2 = Elem("weakest_Yb_Kyoto_MIT_GSI_2022")
    yb2_alphaNP = -1.133363e-9
    assert np.isclose(yb2.alphaNP_GKP(), yb2_alphaNP, atol=0, rtol=1e-7)

    yb3 = Elem("strongest_Yb_Kyoto_MIT_GSI_PTB_2024")
    yb3_alphaNP = 7.49577e-11
    assert np.isclose(yb3.alphaNP_GKP(), yb3_alphaNP, atol=0, rtol=1e-5)

    yb4 = Elem("weakest_Yb_Kyoto_MIT_GSI_PTB_2024")
    yb4_alphaNP = -2.74101e-9
    assert np.isclose(yb4.alphaNP_GKP(), yb4_alphaNP, atol=0, rtol=1e-6)

    vold, vol1, inds = ca.alphaNP_NMGKP_part(3)
    assert len(vold) == len(vol1), (len(vold), len(vol1))


def test_alphaNP_NMGKP():
    ca = Elem('Ca_testdata')
    vold, vol1, inds = ca.alphaNP_NMGKP_part(3)

    assert len(vold) == len(vol1), (len(vold), len(vol1))

    assert np.isclose(ca.alphaNP_NMGKP(), check_alphaNP_NMGKP(ca, 3)[0], atol=0,
        rtol=1e-21)


def test_alphaNP_proj():
    ca = Elem('Ca_testdata')

    assert np.isclose(ca.Fji(j=1, i=0), ca.F1[1], atol=0, rtol=1e-25)
    assert np.isclose(ca.Xji(j=1, i=0), ca.X1[1], atol=0, rtol=1e-25)
    assert np.isclose(
        ca.Xji(j=1, i=0), ca.Xvec[1] - ca.Fji(j=1, i=0) * ca.Xvec[0],
        atol=0, rtol=1e-25)

    F21_64 = (ca.F1[2]).astype(np.float64)
    F21ind_64 = (ca.Fji(j=2, i=0).astype(np.float64))
    assert np.isclose(F21_64, F21ind_64, atol=0, rtol=1e-25)
    assert np.isclose(ca.Fji(j=2, i=0), ca.F1[2], atol=0, rtol=1e-19)
    assert np.isclose(ca.Xji(j=2, i=0), ca.X1[2], atol=0, rtol=1e-25)

    assert np.isclose(ca.alphaNP_proj(), ca.alphaNP_GKP(), atol=0, rtol=20)

    lenp = len(list(product(
        combinations(ca.range_a, 3), combinations(ca.range_i, 2))))

    alphapartlist, xindlist = ca.alphaNP_proj_part(3)

    assert (alphapartlist.shape[0] == lenp)
    assert len(xindlist) == lenp
    for xind in xindlist:
        assert len(xind) == 2

    camin = Elem('Camin')
    alphapartlist, xindlist = ca.alphaNP_proj_part(3)

    assert np.isclose(camin.alphaNP_proj(), camin.alphaNP_GKP(), atol=0, rtol=1)
                      # rtol=20)

    ca24 = Elem("Ca_WT_Aarhus_2024")

    alphaca = ca24.alphaNP_proj(ainds=[0, 1, 2, 3], iinds=[0, 1])
    # pprint("ca24 kifit", alphaca)
    alphaca_Mathematica = 2.45796e-11

    assert np.isclose(alphaca, alphaca_Mathematica, atol=0, rtol=1e-4)


if __name__ == "__main__":
    test_load_all()
    test_load_individual()
    test_linfit()
    test_set_fit_params()
    test_constr_dvec()
    test_swap_elem()
    test_levi_civita()
    test_alphaNP_GKP()
    test_alphaNP_NMGKP()
    test_alphaNP_proj()
