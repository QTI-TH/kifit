import os
import numpy as np
from pprint import pprint
from Mathematica_crosschecks import *
from kifit.builder import Element


user_elems = ['Ca_testdata']
Element.DATA_PATH = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
Element.VALID_ELEM = user_elems


def test_load_all():
    all_element_data = Element.load_all()

    assert np.allclose(
        np.nan_to_num(all_element_data["Ca_testdata"].nu).sum(axis=1),
        [8.44084572e08, 1.68474592e09, 3.38525526e09],
        rtol=1e-8,
    )

    # with nans:
    # assert np.allclose(
    #     np.nan_to_num(all_element_data['Ca_testdata'].nu).sum(axis=1),
    #     [8.44084572e+08, 1.68474592e+09, 7.75849948e+09, 3.38525526e+09],
    #     rtol=1e-8,
    # )
    #


def test_load_individual():
    ca = Element.get("Ca_testdata")
    pprint(ca)

    Ca = Element.get("Ca_testdata")
    assert np.all(np.nan_to_num(ca.nu) == np.nan_to_num(Ca.nu))
    assert np.isclose(np.sum(ca.m_a), 119.88777255, rtol=1e-5)
    assert np.isclose(np.sum(ca.m_ap), 133.86662192999998, rtol=1e-5)
    assert np.isclose(np.sum(ca.sig_m_a**2), 1.452e-15, rtol=1e-5)
    assert np.isclose(np.sum(ca.sig_m_ap**2), 1.15961e-13, rtol=1e-5)

    assert ca.m_nisotopepairs == len(ca.ap_nisotope)
    assert ca.m_nisotopepairs == len(ca.m_a)
    assert ca.m_nisotopepairs == len(ca.m_ap)
    assert ca.m_nisotopepairs == len(ca.nu)
    assert ca.m_nisotopepairs == len(ca.sig_nu)
    assert ca.n_ntransitions == ca.sig_nu.shape[1]

    assert len(ca.m_a) == len(ca.m_ap)
    assert all(u != v for u, v in zip(ca.m_a, ca.m_ap))

    assert len(ca.Xcoeffs) > 0, len(ca.Xcoeffs)
    assert len(ca.sig_Xcoeffs) > 0, len(ca.sig_Xcoeffs)
    assert len(ca.X) == ca.n_ntransitions, len(ca.X)
    assert len(ca.sig_X) == ca.n_ntransitions, len(ca.sig_X)

    for x in range(len(ca.Xcoeffs)):
        assert len(ca.Xcoeffs[x]) == ca.n_ntransitions + 1, len(ca.Xcoeffs[x])
        assert len(ca.sig_Xcoeffs[x]) == ca.n_ntransitions + 1, len(ca.sig_Xcoeffs[x])

    assert ca.nu.size == ca.m_nisotopepairs * ca.n_ntransitions
    assert np.allclose(ca.mu_norm_isotope_shifts, mnu_Mathematica, rtol=1e-14)
    assert np.allclose(ca.sig_mu_norm_isotope_shifts, sig_mnu_Mathematica, rtol=1e-14)
    assert np.sum(ca.corr_nu_nu) == ca.m_nisotopepairs * ca.n_ntransitions
    assert np.sum(ca.corr_m_m) == 1
    assert np.trace(ca.corr_m_mp) == 0
    assert np.trace(ca.corr_mp_mp) == ca.m_nisotopepairs
    assert np.trace(ca.corr_X_X) == ca.n_ntransitions
    assert ca.mu_aap[0] == 1 / ca.m_a[0] - 1 / ca.m_ap[0]
    assert len(ca.mu_aap) == ca.m_nisotopepairs
    assert ca.h_aap.size == ca.m_nisotopepairs
    assert np.allclose(
        ca.h_aap,
        np.array(
            [(ca.a_nisotope[a] - ca.ap_nisotope[a]) / ca.mu_aap[a] for a in ca.range_a]
        ),
        rtol=1e-5,
    )
    assert ca.np_term.size == ca.m_nisotopepairs * ca.n_ntransitions
    assert np.all([i.is_integer() for i in ca.a_nisotope])
    assert np.all([i.is_integer() for i in ca.ap_nisotope])
    assert ca.F1.size == ca.n_ntransitions
    assert ca.F1sq == ca.F1 @ ca.F1
    assert np.isclose(ca.F1[0], 1, rtol=1e-17)
    assert ca.secph1.size == ca.n_ntransitions
    assert np.isclose(ca.secph1[0], 0, rtol=1e-17)
    assert ca.Kperp1.size == ca.n_ntransitions
    assert np.isclose(ca.Kperp1[0], 0, rtol=1e-17)
    assert ca.X1.size == ca.n_ntransitions
    assert np.isclose(ca.X1[0], 0, rtol=1e-17)
    assert sum(ca.range_i) == sum(ca.range_j)
    assert len(ca.range_i) == len(ca.range_j) + 1


def test_set_fit_params():
    ca = Element.get("Ca_testdata")

    kappaperp1temp = list(range(ca.n_ntransitions - 1))
    ph1temp = [np.pi / 2 - np.pi / (i + 2) for i in range(ca.n_ntransitions - 1)]
    alphaNPtemp = 1 / (4 * np.pi)

    ca._update_fit_params([kappaperp1temp, ph1temp, alphaNPtemp])
    assert np.sum(ca.Kperp1) == np.sum(kappaperp1temp)
    assert len(ca.Kperp1) == (len(kappaperp1temp) + 1)
    assert np.sum(ca.ph1) == np.sum(ph1temp)
    assert len(ca.ph1) == len(ph1temp)
    assert ca.alphaNP == alphaNPtemp
    assert all(ca.F1[1:] == np.tan(ph1temp))

    ca._update_fit_params(
        [kappaperp1nit_Mathematica, ph1nit_Mathematica, alphaNP_Mathematica]
    )
    assert np.sum(ca.Kperp1) == np.sum(kappaperp1nit_Mathematica)
    assert np.sum(ca.ph1) == np.sum(ph1nit_Mathematica)
    assert ca.alphaNP == alphaNP_Mathematica
    assert (ca.F1[1:] == np.tan(ph1nit_Mathematica)).all()
    assert np.isclose(ca.F1sq, F1_Mathematica @ F1_Mathematica, rtol=1e-15)

    ca._update_fit_params(
        [kappaperp1nit_LL_Mathematica, ph1nit_LL_Mathematica, alphaNP_Mathematica]
    )
    assert np.sum(ca.Kperp1) == np.sum(kappaperp1nit_LL_Mathematica)
    assert np.sum(ca.ph1) == np.sum(ph1nit_LL_Mathematica)
    assert ca.alphaNP == alphaNP_Mathematica
    assert (ca.F1[1:] == np.tan(ph1nit_LL_Mathematica)).all()
    assert np.isclose(ca.F1sq, F1_LL_Mathematica @ F1_LL_Mathematica, rtol=1e-15)


def test_constr_dvec():
    ca = Element.get("Ca_testdata")

    # without NP
    ca._update_fit_params([kappaperp1nit_LL_Mathematica, ph1nit_LL_Mathematica, 0.0])

    assert (
        np.isclose(spp, spm, rtol=1e-7)
        for (spp, spm) in zip(ca.secph1, secph1nit_LL_Mathematica)
    ), (ca.secph1, secph1nit_LL_Mathematica)
    assert np.allclose(ca.mu_aap, mu_aap_Mathematica, rtol=1e-10)

    D_a1i_python = [[ca.D_a1i(a, i) for i in ca.range_i] for a in ca.range_a]
    assert np.allclose(D_a1i_Mathematica, D_a1i_python, rtol=1e-15)

    assert np.allclose(dmat_Mathematica, ca.dmat, rtol=1e-8)  # can we do better here?

    # with NP
    ca._update_fit_params([kappaperp1nit_LL_Mathematica, ph1nit_LL_Mathematica, 1.0])

    assert np.allclose(ca.np_term, NP_term_alphaNP_1_Mathematica, rtol=1e-14)

    D_a1i_alphaNP_1_python = [[ca.D_a1i(a, i) for i in ca.range_i] for a in ca.range_a]
    assert np.allclose(D_a1i_alphaNP_1_Mathematica, D_a1i_alphaNP_1_python, rtol=1e-15)

    assert np.allclose(dmat_alphaNP_1_Mathematica, ca.dmat, rtol=1e-14)

    absd_explicit = np.array(
        [
            np.sqrt(
                np.sum(np.fromiter([ca.d_ai(a, i) ** 2 for i in ca.range_i], float))
            )
            for a in ca.range_a
        ]
    )
    assert np.allclose(ca.absd, absd_explicit, rtol=1e-25)


def test_cov_nu_nu():
    ca = Element.get("Ca_testdata")
    ca._update_fit_params([kappaperp1nit_LL_Mathematica, ph1nit_LL_Mathematica, 0.0])

    DdDnu_11bj_python = np.array(
        [[ca.DdDnu[0, 0, b, j] for j in ca.range_i] for b in ca.range_a]
    )
    DdDnu_12bj_python = np.array(
        [[ca.DdDnu[0, 1, b, j] for j in ca.range_i] for b in ca.range_a]
    )
    assert np.allclose(DdDnu_11bj_python, DdDnu_11bj_Mathematica)
    assert np.allclose(DdDnu_12bj_python, DdDnu_12bj_Mathematica)

    sigdnu_python = np.einsum("aick,ckdl,bjdl->aibj", ca.DdDnu, ca.cov_nu_nu, ca.DdDnu)

    assert np.allclose(sigdnu_python, sigdnu_Mathematica, rtol=1e-14)
    # (same with and without NP)


def test_cov_m_m():
    ca = Element.get("Ca_testdata")

    # without NP
    ca._update_fit_params([kappaperp1nit_LL_Mathematica, ph1nit_LL_Mathematica, 0.0])

    DdDm_a1b_python = np.array(
        [[ca.DdDm[a, 0, b] for b in ca.range_a] for a in ca.range_a]
    )
    DdDm_a2b_python = np.array(
        [[ca.DdDm[a, 1, b] for b in ca.range_a] for a in ca.range_a]
    )

    assert np.isclose(ca.D_a1i(0, 0), 0, rtol=1e-17)

    assert ca.DdDm[0, 0, 0] == ca.fDdDm_aib(0, 0, 0)
    assert ca.DdDmp[0, 0, 0] == ca.fDdDmp_aib(0, 0, 0)

    for a in ca.range_a:
        assert np.all(DdDm_a1b_python[a] == DdDm_a1b_python[a, 0])
        assert np.isclose(DdDm_a11_Mathematica[a], DdDm_a1b_python[a, 0], rtol=1e-14)
        assert np.all(DdDm_a2b_python[a] == DdDm_a2b_python[a, 0])
        assert np.isclose(DdDm_a21_Mathematica[a], DdDm_a2b_python[a, 0], rtol=1e-14)

    sigdm_python = np.einsum("aic,cd,bjd->aibj", ca.DdDm, ca.cov_m_m, ca.DdDm)
    assert np.allclose(sigdm_python, sigdm_Mathematica, rtol=1e-13)

    # with NP
    ca._update_fit_params([kappaperp1nit_LL_Mathematica, ph1nit_LL_Mathematica, 1.0])

    DdDm_a1b_alphaNP_1_python = np.array(
        [[ca.DdDm[a, 0, b] for b in ca.range_a] for a in ca.range_a]
    )
    DdDm_a2b_alphaNP_1_python = np.array(
        [[ca.DdDm[a, 1, b] for b in ca.range_a] for a in ca.range_a]
    )

    assert ca.DdDm[0, 0, 0] == ca.fDdDm_aib(0, 0, 0)
    assert ca.DdDmp[0, 0, 0] == ca.fDdDmp_aib(0, 0, 0)

    np.allclose(DdDm_a1b_alphaNP_1_python, DdDm_a11_alphaNP_1_Mathematica, rtol=1e-40)

    np.allclose(DdDm_a2b_alphaNP_1_python, DdDm_a21_alphaNP_1_Mathematica, rtol=1e-40)

    for a in ca.range_a:
        assert np.all(DdDm_a1b_alphaNP_1_python[a] == DdDm_a1b_alphaNP_1_python[a, 0])

        assert np.all(DdDm_a2b_alphaNP_1_python[a] == DdDm_a2b_alphaNP_1_python[a, 0])

    sigdm_alphaNP_1_python = np.einsum("aic,cd,bjd->aibj", ca.DdDm, ca.cov_m_m, ca.DdDm)
    assert np.allclose(sigdm_alphaNP_1_python, sigdm_alphaNP_1_Mathematica, rtol=1e-13)


def test_cov_mp_mp():
    ca = Element.get("Ca_testdata")
    ca._update_fit_params([kappaperp1nit_LL_Mathematica, ph1nit_LL_Mathematica, 0.0])

    DdDmp_a1b_python = np.array(
        [[ca.DdDmp[a, 0, b] for b in ca.range_a] for a in ca.range_a]
    )
    DdDmp_a2b_python = np.array(
        [[ca.DdDmp[a, 1, b] for b in ca.range_a] for a in ca.range_a]
    )

    assert np.all(np.diag(np.diag(DdDmp_a1b_python)) == DdDmp_a1b_python)
    assert np.allclose(DdDmp_a1a_Mathematica, np.diag(DdDmp_a1b_python))

    assert np.all(np.diag(np.diag(DdDmp_a2b_python)) == DdDmp_a2b_python)
    assert np.allclose(DdDmp_a2a_Mathematica, np.diag(DdDmp_a2b_python))

    sigdmp_python = np.einsum("aic,cd,bjd->aibj", ca.DdDmp, ca.cov_mp_mp, ca.DdDmp)
    assert np.allclose(sigdmp_python, sigdmp_Mathematica, rtol=1e-13)


def test_cov_m_mp():
    ca = Element.get("Ca_testdata")
    ca._update_fit_params([kappaperp1nit_LL_Mathematica, ph1nit_LL_Mathematica, 0.0])

    sigdmmp_python = np.einsum("aic,cd,bjd->aibj", ca.DdDm, ca.cov_m_mp, ca.DdDmp)

    assert not ca.cov_m_mp.any()
    assert not sigdmmp_python.any()


def test_cov_X_X():
    ca = Element.get("Ca_testdata")
    ca._update_fit_params([kappaperp1nit_LL_Mathematica, ph1nit_LL_Mathematica, 0.0])

    assert np.isclose(ca.sig_X @ ca.sig_X, np.sum(ca.cov_X_X), rtol=1e-25)
    assert np.allclose(ca.X / 10, ca.sig_X, rtol=1e-25)

    # with NP
    ca._update_fit_params([kappaperp1nit_LL_Mathematica, ph1nit_LL_Mathematica, 1.0])

    assert np.allclose(ca.DdDX[0], DdDX_1ij_alphaNP_1_Mathematica, rtol=1e-25)

    sigdX_python = np.einsum("aic,cd,bjd->aibj", ca.DdDX, ca.cov_X_X, ca.DdDX)
    assert np.allclose(sigdX_python, sigdX_Mathematica, rtol=1e-14)


def test_constr_LL():
    ca = Element.get("Ca_testdata")
    ca._update_fit_params([kappaperp1nit_LL_Mathematica, ph1nit_LL_Mathematica, 0.0])

    assert np.allclose(ca.absd, absd_Mathematica, rtol=1e-9)

    normald_python = ca.dmat / ca.absd[:, None]
    assert np.allclose(normald_python, normald_Mathematica, rtol=1e-25)

    assert np.allclose(ca.cov_d_d, cov_d_d_Mathematica, rtol=1e-8)
    assert np.allclose(
        np.linalg.inv(ca.cov_d_d), np.linalg.inv(cov_d_d_Mathematica), rtol=1e-25
    )
    assert np.allclose(np.linalg.inv(ca.cov_d_d), inv_cov_d_d_Mathematica, rtol=1e-25)
    assert np.allclose(
        ca.cov_d_d @ np.linalg.inv(ca.cov_d_d),
        np.identity(ca.m_nisotopepairs),
        rtol=1e-30,
    )
    assert np.allclose(
        np.linalg.inv(ca.cov_d_d) @ ca.cov_d_d,
        np.identity(ca.m_nisotopepairs),
        rtol=1e-30,
    )

    assert np.isclose(ca.LL, LL_Mathematica, rtol=1e-25)

    simplified_LL = 1 / 2 * np.sum(ca.absd**2 / np.diag(ca.cov_d_d))

    assert np.isclose(simplified_LL, ca.LL, rtol=1e-25)

    ca._update_fit_params([kappaperp1nit_LL_Mathematica, ph1nit_LL_Mathematica, 1.0])

    assert np.allclose(ca.cov_d_d, cov_d_d_alphaNP_1_Mathematica, rtol=1e-8)
    assert np.allclose(
        np.linalg.inv(ca.cov_d_d),
        np.linalg.inv(cov_d_d_alphaNP_1_Mathematica),
        rtol=1e-25,
    )
    assert np.allclose(
        np.linalg.inv(ca.cov_d_d), inv_cov_d_d_alphaNP_1_Mathematica, rtol=1e-25
    )
    assert np.allclose(
        ca.cov_d_d @ np.linalg.inv(ca.cov_d_d),
        np.identity(ca.m_nisotopepairs),
        atol=1e-3,
    )
    assert np.allclose(
        np.linalg.inv(ca.cov_d_d) @ ca.cov_d_d,
        np.identity(ca.m_nisotopepairs),
        atol=1e-3,
    )

    assert np.isclose(ca.LL, LL_alphaNP_1_Mathematica, rtol=1e-3)


if __name__ == "__main__":
    test_load_all()
    test_load_individual()
    test_set_fit_params()
    test_constr_dvec()
    test_cov_nu_nu()
    test_cov_m_m()
    test_cov_mp_mp()
    test_cov_m_mp()
    test_cov_X_X()
    test_constr_LL()
