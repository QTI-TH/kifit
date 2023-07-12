import numpy as np
from pprint import pprint
from qiss.loadelems import Elem

# kappa1nit_Mathematica = [
#    4.9682114488512552396556008921397693278482513553080 * 1e8,
#    -1.4105967590620197257120218754592295249756731139951 * 1e12,
#    -1.4241471060357918255690206870104588861297809071035 * 1e12
# ]

kappa1nit_Mathematica = [
    4.9742388970274519920349121093750000000000000000000 * 1e8,
    -1.4131282963819233398437500000000000000000000000000 * 1e12,
    -1.4223106941423833007812500000000000000000000000000 * 1e12
]

# kappaperp1nit_Mathematica = [
#    4.9682059838664850121518080332414179970179536048136 * 1e8,
#    -1.1233829500712454612814335159131937488307375126787 * 1e12,
#    -1.3860042013653358138656350500522641761691026897284 * 1e12
# ]

# ph1nit_Mathematica = [
#     -0.0014832333435150817129318183433126588923916195379320,
#     0.64949645146536797653313508057026886658732822381497,
#     -0.23196291438488159187037693718542411633042383666814
# ]

ph1nit_Mathematica = [
    -0.0014834864833783716238368999285057725501246750354767,
    0.65017047287571183566967647493584081530570983886719,
    -0.23269327293226219066646365263295592740178108215332
]

ph1nit1_Mathematica = ph1nit_Mathematica[:]
ph1nit1_Mathematica.insert(0, 0.)

# Kperp1nit_from_Kperp1_Mathematica
kappaperp1nit_Mathematica = [np.cos(kpvars[1]) * kpvars[0] for kpvars in
        zip(kappa1nit_Mathematica, ph1nit1_Mathematica)]

F1_Mathematica = np.insert(np.tan(ph1nit_Mathematica), 0, 1.)
secph1nit_Mathematica = np.insert(1 / np.cos(ph1nit_Mathematica), 0, 0.)

alphaNP_Mathematica = 0.
mu_aa2_Mathematica = 0.004169442382310047

D_21i_Mathematica = [
    0.,
    2.073997339997269255955497341695989353482293625354e6,
    -5.892030917536769653330030299682489425100819176865e9,
    -5.9301874787701413843637317976603758539444923678601e9
]

dmat_Mathematica = [
    [
        2.24301855109825091949720999615246811983265226 * 1e7,
        -24224.850398270659132400280368355607577145434,
        -2.2074790634343157766395429128114598312858862 * 1e7,
        2.38112281035859225321073258866502291329584224 * 1e7
    ],
    [
        -3.25904332142547751892260744844295005506057579 * 1e7,
        35198.031199156676627365078236867570062176001,
        3.2074054399087535881282258562295919188544396 * 1e7,
        -3.45970494840664684316890726393176978971438777 * 1e7,
    ],
    [
        1.01602476518989656095675188588463112614294830 * 1e7,
        -10973.180724302827258962732241150796698230465,
        -9.999263693321154532805171976146030144855092 * 1e6,
        1.07858213928949854358787515678257678244294463 * 1e7
    ]
]

LL_Mathematica = -3.8885010863615294971452723132917170227148523


def test_load_all():
    all_element_data = Elem.load_all()

    assert np.allclose(
        np.nan_to_num(all_element_data['Ca'].nu).sum(axis=1),
        [8.44084572e+08, 1.68474592e+09, 3.38525526e+09],
        rtol=1e-8,
    )

    # with nans:
    # assert np.allclose(
    #     np.nan_to_num(all_element_data['Ca'].nu).sum(axis=1),
    #     [8.44084572e+08, 1.68474592e+09, 7.75849948e+09, 3.38525526e+09],
    #     rtol=1e-8,
    # )
    #


def test_load_individual():
    ca = Elem.get('Ca')
    pprint(ca)

    Ca = Elem.get('Ca')
    assert (np.nan_to_num(ca.nu) == np.nan_to_num(Ca.nu)).all()
    assert np.isclose(np.sum(ca.m_a), 119.88777255, rtol=1e-5)
    assert np.isclose(np.sum(ca.m_ap), 133.86662192999998, rtol=1e-5)
    assert np.isclose(np.sum(ca.sig_m_a), 1.452e-15, rtol=1e-5)
    assert np.isclose(np.sum(ca.sig_m_ap), 1.15961e-13, rtol=1e-5)
    assert (ca.nu.size == ca.m_nisotopepairs * ca.n_ntransitions)
    assert (np.sum(ca.corr_nu_nu) == ca.m_nisotopepairs * ca.n_ntransitions)
    assert (np.trace(ca.corr_m_m) == ca.m_nisotopepairs)
    assert (np.trace(ca.corr_m_mp) == 0)
    assert (np.trace(ca.corr_mp_mp) == ca.m_nisotopepairs)
    assert (np.trace(ca.corr_X_X) == ca.n_ntransitions)
    assert (ca.mu_aap[0] == 1 / ca.m_a[0] - 1 / ca.m_ap[0])
    assert (len(ca.mu_aap) == ca.m_nisotopepairs)
    assert (ca.h_aap.size == ca.m_nisotopepairs)
    assert (ca.np_term.size == ca.m_nisotopepairs * ca.n_ntransitions)
    assert all([i.is_integer() for i in ca.a_nisotope])
    assert all([i.is_integer() for i in ca.ap_nisotope])
    assert (ca.F1.size == ca.n_ntransitions)
    assert np.isclose(ca.F1[0], 1, rtol=1e-17)
    assert (ca.secph1.size == ca.n_ntransitions)
    assert np.isclose(ca.secph1[0], 0, rtol=1e-17)
    assert (ca.Kperp1.size == ca.n_ntransitions)
    assert np.isclose(ca.Kperp1[0], 0, rtol=1e-17)
    assert (ca.X1.size == ca.n_ntransitions)
    assert np.isclose(ca.X1[0], 0, rtol=1e-17)
    assert (sum(ca.range_i) == sum(ca.range_j))
    assert (len(ca.range_i) == len(ca.range_j) + 1)


def test_set_fit_params():
    ca = Elem.get('Ca')

    kappaperp1temp = list(range(ca.n_ntransitions - 1))
    ph1temp = [np.pi / 2 - np.pi / (i + 2) for i in
        range(ca.n_ntransitions - 1)]
    alphaNPtemp = 1 / (4 * np.pi)
    ca._update_fit_params([kappaperp1temp, ph1temp, alphaNPtemp])

    print("sum Kperp python  ", sum(ca.Kperp1))
    print("sum kappaperp1temp", np.sum(kappaperp1temp))

    assert (np.sum(ca.Kperp1) == np.sum(kappaperp1temp))
    assert (len(ca.Kperp1) == (len(kappaperp1temp) + 1))
    assert (np.sum(ca.ph1) == np.sum(ph1temp))
    assert (len(ca.ph1) == len(ph1temp))
    assert (ca.alphaNP == alphaNPtemp)
    assert (ca.F1[1:] == np.tan(ph1temp)).all()

    ca._update_fit_params([kappaperp1nit_Mathematica, ph1nit_Mathematica,
        alphaNP_Mathematica])

    print("sum Kperp python  ", sum(ca.Kperp1))
    print("sum kappaperp1temp", np.sum(kappaperp1nit_Mathematica))

    assert (np.sum(ca.Kperp1) == np.sum(kappaperp1nit_Mathematica))
    assert (np.sum(ca.ph1) == np.sum(ph1nit_Mathematica))
    assert (ca.alphaNP == alphaNP_Mathematica)
    assert (ca.F1[1:] == np.tan(ph1nit_Mathematica)).all()


def test_constr_dvec():
    ca = Elem.get('Ca')
    ca._update_fit_params([kappaperp1nit_Mathematica, ph1nit_Mathematica, 0.])

    # print("Kperp1nit_from_Kperp1_Mathematica", Kperp1nit_from_Kperp1_Mathematica)
    # print("Kperp1nit_Mathematica", Kperp1nit_Mathematica)
    for i in ca.range_i:
        assert (ca.F1[i] == F1_Mathematica[i]), (ca.F1[i], F1_Mathematica[i])
        assert (ca.secph1[i] == secph1nit_Mathematica[i]), (ca.secph1[i],
                secph1nit_Mathematica[i])
    assert (ca.F1sq == F1_Mathematica @ F1_Mathematica)

    for j in ca.range_j:
        assert (ca.Kperp1[j] == kappaperp1nit_Mathematica[j - 1])

    assert (np.isclose(spp, spm, rtol=1e-7) for (spp, spm) in zip(ca.secph1,
        secph1nit_Mathematica)), (ca.secph1, secph1nit_Mathematica)
    assert np.isclose(ca.mu_aap[2], mu_aa2_Mathematica, rtol=1e-10)
    D_21i_python = [ca.D_a1i(2, i) for i in ca.range_i]
    assert (np.isclose(dp, dm, rtol=1e-11) for (dp, dm) in zip(D_21i_python,
        D_21i_Mathematica))
    print("new d_ai(0,0) python", ca.d_ai(0, 0))
    print("fit params sum Kperp python  ", sum(ca.Kperp1))
    print("fit params sum kappaperp1temp", np.sum(kappaperp1nit_Mathematica))

    print("fit params sum ph1 python        ", sum(ca.ph1))
    print("fit params sum ph1nit_Mathematica", sum(ph1nit1_Mathematica))

    # print("secph1 Mathematica", np.insert(1 / np.cos(ph1nit_Mathematica), 0, 0))
    # print("secph1 python", ca.secph1)
    # print("F1 python     ", ca.F1)
    # print("F1_Mathematica", F1_Mathematica)
    # print("F1sq python     ", ca.F1sq)
    # print("F1sq Mathematica", F1_Mathematica @ F1_Mathematica)
    print("d_ai(0,0) test:    ",
        (- 1 / ca.F1sq * np.sum(np.array([ca.F1[j]
            * (ca.D_a1i(0, j) / ca.mu_aap[0]
                - ca.secph1[j] * ca.Kperp1[j]) for j in ca.range_j]))))
    print("d_ai(0,0) eval:    ", ca.d_ai(0, 0))
    print("d_ai(0,0) Mathematica", dmat_Mathematica[0][0])
    print("before d evaluation:")
    print("F1", ca.F1)
    print("Kperp1", ca.Kperp1)
    print("D_a1i(0,0)", ca.D_a1i(0, 0))
    print("mu_aap", ca.mu_aap)
    print("F1sq", ca.F1sq)

    print("D_21i_Mathematica", D_21i_Mathematica)
    print("D_21i python     ", [ca.D_a1i(2, i) for i in ca.range_i])
    # print("kappaperp1nit_Mathematica", kappaperp1nit_Mathematica)
    # print("Kperp1 python        ", ca.Kperp1)
    # for a in ca.range_a:
    #     for i in ca.range_i:
    #
    # print("ca.d_ai(" + str(a) + str(i) +")         ", ca.d_ai(a, i))
    # print("dmat_Mathematica(" + str(a) + str(i) + ")", dmat_Mathematica[a][i])

    for s in range(5):
        assert (np.isclose(spp, spm, rtol=1e-7) for (spp, spm) in zip(ca.secph1,
            secph1nit_Mathematica)), (ca.secph1, secph1nit_Mathematica)

        for a in ca.range_a:
            for i in ca.range_i:
                pass
                # assert (np.isclose(ca.d_ai(a, i), dmat_Mathematica[a][i],
                #     rtol=1e-9))
    # print("dim(cov_nu_nu)", (np.diag(ca.sig_nu) @ ca.corr_nu_nu @ np.diag(ca.sig_nu)))


def test_constr_covdd():
    ca = Elem.get('Ca')
    assert np.isclose(ca.D_a1i(0, 0), 0, rtol=1e-17)
    print("fDdDm_aib", ca.fDdDm_aib(0, 0, 0))
    print("fDdDmp_aib", ca.fDdDmp_aib(0, 0, 0))
    assert ((ca.fDdDm_aib(0, 0, 0) * ca.m_a[0]**2) == (ca.fDdDmp_aib(0, 0, 0)
        * ca.m_ap[0]**2))
    print("m_a", ca.m_a)
    print("m_ap", ca.m_ap)

    print("DdDm", ca.DdDm[0, 0, 0])
    assert (ca.DdDm[0, 0, 0] == ca.fDdDm_aib(0, 0, 0))
    assert (ca.DdDmp[0, 0, 0] == ca.fDdDmp_aib(0, 0, 0))


def test_constr_LL():
    ca = Elem.get('Ca')
    ca._update_fit_params([kappaperp1nit_Mathematica, ph1nit_Mathematica, 0.])
    print("LL_Ca", ca.LL_elem())
    print("LL_Ca_Mathematica", LL_Mathematica)
    print("d[0,0]", ca.d_ai(0, 0))
    print("d00", - 1 / ca.F1sq * np.sum(np.array([ca.F1[j]
                * (- ca.secph1[j] * ca.Kperp1[j]) for j in ca.range_j])))
    print("F1sq", ca.F1sq)
    print("F1", ca.F1)
    print("F1 res", np.insert(np.tan(ca.ph1), 0, 1.))
    # LL ~ (check ll_in_3_steps.wl)

# def test_DdDnu_aibj(self, a: int, i: int, b: int, j: int):
#      """
#      Returns derivative of nu_i^a wrt. nu_j^b,
#      where a = AA', b = BB'.
#
#      """
#      if ((i == 0) & (j == 0) & (a == b) & (a in self.range_a)):
#          return (1 / self.F1sq * 1 / self.mu_aap[a]
#                  * np.sum(np.array([np.tan(self.ph1[j])**2
#                      for j in self.range_j])))
#
#      elif (((i == 0) & (j in self.range_j) & (a == b) & (a in self.range_a))
#              or ((i in self.range_j) & (j == 0) & (a == b) & (a in self.range_a))):
#          return -1 / self.F1sq * 1 / self.mu_aap[a] * np.tan(self.ph1[j])
#
#      elif ((i in self.range_j) & (j in self.range_j) & (a == b)
#              & (a in self.range_a)):
#          res = -1 / self.F1sq * self.F1[i] * self.F1[j]
#
#          if i == j:
#              res += 1
#
#          return res / self.mu_aap[a]
#
#      else:
#          return 0
#


if __name__ == "__main__":
    test_load_individual()
    test_set_fit_params()
    test_constr_dvec()
    # test_constr_covdd()
    # test_constr_LL()

# def export_reduced_isotope_shifts():
#     for elem in ['Ca']:
#         elem_data = Elem.get('Ca')
#         np.savetxt('test_output/mnu_{}.dat'.format(elem), elem_data.mnu, delimiter=',')
#
