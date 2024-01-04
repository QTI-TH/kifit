import numpy as np
from pprint import pprint
from kifit.loadelems import Elem
from Mathematica_crosschecks import *


def test_load_all():
    all_element_data = Elem.load_all()

    assert np.allclose(
        np.nan_to_num(all_element_data['Ca_testdata'].nu).sum(axis=1),
        [8.44084572e+08, 1.68474592e+09, 3.38525526e+09],
        rtol=1e-8,
    )


def test_load_individual():
    ca = Elem.get('Ca_testdata')
    pprint(ca)

    Ca = Elem.get('Ca_testdata')
    assert np.all(np.nan_to_num(ca.nu) == np.nan_to_num(Ca.nu))
    assert np.isclose(np.sum(ca.m_a), 119.88777255, rtol=1e-5)
    assert np.isclose(np.sum(ca.m_ap), 133.86662192999998, rtol=1e-5)
    assert np.isclose(np.sum(ca.sig_m_a**2), 1.452e-15, rtol=1e-5)
    assert np.isclose(np.sum(ca.sig_m_ap**2), 1.15961e-13, rtol=1e-5)

    assert ca.nisotopepairs == len(ca.ap_nisotope)
    assert ca.nisotopepairs == len(ca.m_a)
    assert ca.nisotopepairs == len(ca.m_ap)
    assert ca.nisotopepairs == len(ca.nu)
    assert ca.nisotopepairs == len(ca.sig_nu)
    assert ca.ntransitions == ca.sig_nu.shape[1]

    assert len(ca.m_a) == len(ca.m_ap)
    assert all(u != v for u, v in zip(ca.m_a, ca.m_ap))

    assert (len(ca.Xcoeffs) > 0), len(ca.Xcoeffs)
    assert (len(ca.sig_Xcoeffs) > 0), len(ca.sig_Xcoeffs)
    assert len(ca.X) == ca.ntransitions, len(ca.X)
    assert len(ca.sig_X) == ca.ntransitions, len(ca.sig_X)

    for x in range(len(ca.Xcoeffs)):
        assert len(ca.Xcoeffs[x]) == ca.ntransitions + 1, len(ca.Xcoeffs[x])
        assert len(ca.sig_Xcoeffs[x]) == ca.ntransitions + 1, len(ca.sig_Xcoeffs[x])

    assert (ca.nu.size == ca.nisotopepairs * ca.ntransitions)
    assert np.allclose(ca.mu_norm_isotope_shifts, mnu_Mathematica, rtol=1e-14)
    assert np.allclose(ca.sig_mu_norm_isotope_shifts, sig_mnu_Mathematica,
            rtol=1e-14)
    # assert (np.sum(ca.corr_nu_nu) == ca.nisotopepairs * ca.ntransitions)
    # assert (np.sum(ca.corr_m_m) == 1)
    # assert (np.trace(ca.corr_m_mp) == 0)
    # assert (np.trace(ca.corr_mp_mp) == ca.nisotopepairs)
    # assert (np.trace(ca.corr_X_X) == ca.ntransitions)
    assert (ca.mu_aap[0] == 1 / ca.m_a[0] - 1 / ca.m_ap[0])
    assert (len(ca.mu_aap) == ca.nisotopepairs)
    assert (ca.h_aap.size == ca.nisotopepairs)
    assert np.allclose(ca.h_aap, np.array([(ca.a_nisotope[a] - ca.ap_nisotope[a])
        / ca.mu_aap[a] for a in ca.range_a]), rtol=1e-5)
    assert (ca.np_term.size == ca.nisotopepairs * ca.ntransitions)
    assert np.all([i.is_integer() for i in ca.a_nisotope])
    assert np.all([i.is_integer() for i in ca.ap_nisotope])
    assert (ca.F1.size == ca.ntransitions)
    assert (ca.F1sq == ca.F1 @ ca.F1)
    assert np.isclose(ca.F1[0], 1, rtol=1e-17)
    assert (ca.secph1.size == ca.ntransitions)
    assert np.isclose(ca.secph1[0], 0, rtol=1e-17)
    assert (ca.Kperp1.size == ca.ntransitions)
    assert np.isclose(ca.Kperp1[0], 0, rtol=1e-17)
    assert (ca.X1.size == ca.ntransitions)
    assert np.isclose(ca.X1[0], 0, rtol=1e-17)
    assert (sum(ca.range_i) == sum(ca.range_j))
    assert (len(ca.range_i) == len(ca.range_j) + 1)


def test_set_fit_params():
    ca = Elem.get('Ca_testdata')

    kappaperp1temp = np.array(list(range(ca.ntransitions - 1)))
    ph1temp = np.array([np.pi / 2 - np.pi / (i + 2) for i in
        range(ca.ntransitions - 1)])
    alphaNPtemp = 1 / (4 * np.pi)

    sigkappeperp1temp = 1e-5 * kappaperp1temp
    sigph1temp = 1e-3 * ph1temp
    sigalphaNPtemp = 0

    ca._update_fit_params([kappaperp1temp, ph1temp, alphaNPtemp],
            [sigkappeperp1temp, sigph1temp, sigalphaNPtemp])
    assert (np.sum(ca.kp1) == np.sum(kappaperp1temp))
    assert (np.sum(ca.Kperp1) == np.sum(kappaperp1temp))
    assert (len(ca.Kperp1) == (len(kappaperp1temp) + 1))
    assert (np.sum(ca.ph1) == np.sum(ph1temp))
    assert (len(ca.ph1) == len(ph1temp))
    assert (ca.alphaNP == alphaNPtemp)
    assert all(ca.F1[1:] == np.tan(ph1temp))
    assert (np.sum(ca.sig_kp1) == np.sum(sigkappeperp1temp))
    assert (np.sum(ca.sig_ph1) == np.sum(sigph1temp))
    assert (ca.sig_alphaNP == sigalphaNPtemp)

    sig_kappaperp1nit_Mathematica = np.zeros(ca.ntransitions - 1)
    sig_ph1nit_Mathematica = np.zeros(ca.ntransitions - 1)

    ca._update_fit_params(
        [kappaperp1nit_Mathematica, ph1nit_Mathematica, alphaNP_Mathematica],
        [sig_kappaperp1nit_Mathematica, sig_ph1nit_Mathematica, 0.])
    assert (np.sum(ca.Kperp1) == np.sum(kappaperp1nit_Mathematica))
    assert (np.sum(ca.kp1) == np.sum(kappaperp1nit_Mathematica))
    assert (np.sum(ca.ph1) == np.sum(ph1nit_Mathematica))
    assert (ca.alphaNP == alphaNP_Mathematica)
    assert (ca.F1[1:] == np.tan(ph1nit_Mathematica)).all()
    assert np.isclose(ca.F1sq, F1_Mathematica @ F1_Mathematica, rtol=1e-15)
    assert (np.sum(ca.sig_kp1) == np.sum(sig_kappaperp1nit_Mathematica))
    assert (np.sum(ca.sig_ph1) == np.sum(sig_ph1nit_Mathematica))
    assert (ca.sig_alphaNP == 0.)


    ca._update_fit_params(
        [kappaperp1nit_LL_Mathematica, ph1nit_LL_Mathematica, 0.],
        [sig_kappaperp1nit_Mathematica, sig_ph1nit_Mathematica, 0.])

    assert (np.sum(ca.Kperp1) == np.sum(kappaperp1nit_LL_Mathematica))
    assert (np.sum(ca.ph1) == np.sum(ph1nit_LL_Mathematica))
    assert (ca.alphaNP == alphaNP_Mathematica)
    assert (ca.F1[1:] == np.tan(ph1nit_LL_Mathematica)).all()
    assert np.isclose(ca.F1sq, F1_LL_Mathematica @ F1_LL_Mathematica, rtol=1e-15)


def test_constr_dvec():
    ca = Elem.get('Ca_testdata')

    sig_kappaperp1nit_Mathematica = np.zeros(ca.ntransitions - 1)
    sig_ph1nit_Mathematica = np.zeros(ca.ntransitions - 1)

    # without NP
    ca._update_fit_params(
        [kappaperp1nit_LL_Mathematica, ph1nit_LL_Mathematica, 0.],
        [sig_kappaperp1nit_Mathematica, sig_ph1nit_Mathematica, 0.])

    assert (np.isclose(spp, spm, rtol=1e-7) for (spp, spm) in zip(ca.secph1,
        secph1nit_LL_Mathematica)), (ca.secph1, secph1nit_LL_Mathematica)
    assert np.allclose(ca.mu_aap, mu_aap_Mathematica, rtol=1e-10)

    D_a1i_python = [[ca.D_a1i(a, i) for i in ca.range_i] for a in ca.range_a]
    assert np.allclose(D_a1i_Mathematica, D_a1i_python, rtol=1e-15)

    assert np.allclose(dmat_Mathematica, ca.dmat, rtol=1e-8)   # can we do better here?

    # with NP
    ca._update_fit_params(
        [kappaperp1nit_LL_Mathematica, ph1nit_LL_Mathematica, 1.],
        [sig_kappaperp1nit_Mathematica, sig_ph1nit_Mathematica, 0.])

    assert np.allclose(ca.np_term, NP_term_alphaNP_1_Mathematica, rtol=1e-14)

    D_a1i_alphaNP_1_python = [[ca.D_a1i(a, i) for i in ca.range_i] for a in ca.range_a]
    assert np.allclose(D_a1i_alphaNP_1_Mathematica, D_a1i_alphaNP_1_python, rtol=1e-15)

    assert np.allclose(dmat_alphaNP_1_Mathematica, ca.dmat, rtol=1e-14)

    absd_explicit = np.array([np.sqrt(np.sum(
        np.fromiter([ca.d_ai(a, i)**2 for i in ca.range_i], float))) for a in ca.range_a])
    assert np.allclose(ca.absd, absd_explicit, rtol=1e-25)


if __name__ == "__main__":
    test_load_all()
    test_load_individual()
    test_set_fit_params()
    test_constr_dvec()
