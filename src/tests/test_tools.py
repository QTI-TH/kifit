import numpy as np
from itertools import product, combinations

from kifit.build import Elem, perform_odr, perform_linreg
from kifit.detools import (
    sample_gkp_parts, assemble_gkp_combinations,
    sample_proj_parts, assemble_proj_combinations
)

np.random.seed(1)

# import matplotlib.pyplot as plt


def test_linfit():

    ca = Elem('Ca_testdata')

    (betas_odr, sig_betas_odr, kperp1s_odr, ph1s_odr,
        sig_kperp1s_odr, sig_ph1s_odr, cov_kperp1_ph1s) = perform_odr(
        ca.nutil_in, ca.sig_nutil_in,
        reference_transition_index=0)

    (betas_linreg, sig_betas_linreg, kperp1s_linreg, ph1s_linreg,
        sig_kperp1s_linreg, sig_ph1s_linreg) = perform_linreg(
        ca.nutil_in, reference_transition_index=0)

    assert betas_odr.shape == (ca.ntransitions - 1, 2)
    assert betas_linreg.shape == (ca.ntransitions - 1, 2)

    assert np.all(np.isclose(betas_odr, betas_linreg, atol=0, rtol=1e-2))
    assert np.all(np.isclose(sig_betas_odr, sig_betas_linreg, atol=0, rtol=1))
    assert np.all(np.isclose(kperp1s_odr, kperp1s_linreg, atol=0, rtol=1e-2))
    assert np.all(np.isclose(ph1s_odr, ph1s_linreg, atol=0, rtol=1e-2))
    assert np.all(np.isclose(sig_kperp1s_odr, sig_kperp1s_linreg, atol=0, rtol=1))
    assert np.all(np.isclose(sig_ph1s_odr, sig_ph1s_linreg, atol=0, rtol=1))

    xvals = ca.nutil_in.T[0]
    yvals = ca.nutil_in[:, 1:]

    betas_dat = np.array([np.polyfit(xvals, yvals[:, i], 1) for i in
        range(yvals.shape[1])])

    assert betas_dat.shape == (ca.ntransitions -1, 2)
    assert np.all(np.isclose(betas_dat, betas_odr, atol=0, rtol=1e-2))


def test_GKP_combinations():

    camin = Elem("Camin")

    dim_gkp = 3

    (
        meanvd_min, sigvd_min, meanv1_min, sigv1_min, xindlist_min
    ) = sample_gkp_parts(camin, 1, dim_gkp, "gkp")
    # one sample so no uncertainty computed

    alphas_min, sigalphas_min = assemble_gkp_combinations(
        camin, meanvd_min, sigvd_min, meanv1_min, sigv1_min, xindlist_min,
        dim_gkp, "gkp")

    lenp_gkp = len(list(
        product(
            combinations(camin.range_a, dim_gkp),
            combinations(camin.range_i, dim_gkp - 1))))

    assert alphas_min.shape[0] == lenp_gkp
    assert sigalphas_min.shape[0] == lenp_gkp

    assert np.isclose(camin.alphaNP_GKP(), alphas_min[0], atol=0, rtol=1e-13)

    assert np.isclose(alphas_min[0], camin.alphaNP_proj_combinations(3)[0],
        atol=0, rtol=1)  # rtol=1e-13)

    (
        meanvd, sigvd, meanv1, sigv1, xindlist
    ) = sample_gkp_parts(camin, 10000, dim_gkp, "gkp")

    alphas, sigalphas = assemble_gkp_combinations(
        camin, meanvd, sigvd, meanv1, sigv1, xindlist,
        dim_gkp, "gkp")

    # check this
    # print("alphas[0]  ", alphas[0])
    # print("alphaNP_GKP", camin.alphaNP_GKP())

    assert np.isclose(camin.alphaNP_GKP(), alphas[0], atol=0, rtol=3)

    print("sigalphas gkp camin", sigalphas)

    yb1 = Elem("strongest_Yb_Kyoto_MIT_GSI_2022")

    vd_yb1, sig_vd_yb1, v1_yb1, sig_v1_yb1, xinds_yb1 = sample_gkp_parts(
        elem=yb1,
        nsamples=1000,
        dim=3,
        detstr="gkp")

    alphaNPs_yb1, sig_alphaNPs_yb1 = assemble_gkp_combinations(
        elem=yb1,
        meanvoldat=vd_yb1,
        sigvoldat=sig_vd_yb1,
        meanvol1=v1_yb1,
        sigvol1=sig_v1_yb1,
        xindlist=xinds_yb1,
        dim=3,
        detstr="gkp")

    yb1_UB = 9.018604e-11

    assert np.isclose(np.abs(alphaNPs_yb1[0]) + 2 * sig_alphaNPs_yb1[0], yb1_UB,
        atol=0, rtol=1e-1)

    yb2 = Elem("weakest_Yb_Kyoto_MIT_GSI_2022")

    vd_yb2, sig_vd_yb2, v1_yb2, sig_v1_yb2, xinds_yb2 = sample_gkp_parts(
        elem=yb2,
        nsamples=1000,
        dim=3,
        detstr="gkp")

    alphaNPs_yb2, sig_alphaNPs_yb2 = assemble_gkp_combinations(
        elem=yb2,
        meanvoldat=vd_yb2,
        sigvoldat=sig_vd_yb2,
        meanvol1=v1_yb2,
        sigvol1=sig_v1_yb2,
        xindlist=xinds_yb2,
        dim=3,
        detstr="gkp")

    yb2_UB = 4.423329e-9

    assert np.isclose(alphaNPs_yb2[0] + 2 * sig_alphaNPs_yb2[0], yb2_UB,
        atol=0, rtol=1)

    yb3 = Elem("strongest_Yb_Kyoto_MIT_GSI_PTB_2024")

    vd_yb3, sig_vd_yb3, v1_yb3, sig_v1_yb3, xinds_yb3 = sample_gkp_parts(
        elem=yb3,
        nsamples=1000,
        dim=3,
        detstr="gkp")

    alphaNPs_yb3, sig_alphaNPs_yb3 = assemble_gkp_combinations(
        elem=yb3,
        meanvoldat=vd_yb3,
        sigvoldat=sig_vd_yb3,
        meanvol1=v1_yb3,
        sigvol1=sig_v1_yb3,
        xindlist=xinds_yb3,
        dim=3,
        detstr="gkp")

    yb3_UB = 1.24674e-10

    assert np.isclose(np.abs(alphaNPs_yb3[0]) + 2 * sig_alphaNPs_yb3[0], yb3_UB,
        atol=0, rtol=1)

    yb4 = Elem("weakest_Yb_Kyoto_MIT_GSI_PTB_2024")

    vd_yb4, sig_vd_yb4, v1_yb4, sig_v1_yb4, xinds_yb4 = sample_gkp_parts(
        elem=yb4,
        nsamples=1000,
        dim=3,
        detstr="gkp")

    alphaNPs_yb4, sig_alphaNPs_yb4 = assemble_gkp_combinations(
        elem=yb4,
        meanvoldat=vd_yb4,
        sigvoldat=sig_vd_yb4,
        meanvol1=v1_yb4,
        sigvol1=sig_v1_yb4,
        xindlist=xinds_yb4,
        dim=3,
        detstr="gkp")

    yb4_UB = 9.9457e-9

    assert np.isclose(np.abs(alphaNPs_yb4[0]) + 2 * sig_alphaNPs_yb4[0], yb4_UB,
        atol=0, rtol=1)


def test_NMGKP_combinations():

    camin = Elem("Ca_testdata")
    dim_nmgkp = 3

    vd_min, sig_vd_min, v1_min, sig_v1_min, xinds_min = sample_gkp_parts(
        elem=camin,
        nsamples=1,
        dim=dim_nmgkp,
        detstr="nmgkp")
    # one sample so no uncertainty computed

    alphas_min, sigalphas_min = assemble_gkp_combinations(
        elem=camin,
        meanvoldat=vd_min,
        sigvoldat=sig_vd_min,
        meanvol1=v1_min,
        sigvol1=sig_v1_min,
        xindlist=xinds_min,
        dim=3,
        detstr="nmgkp")
    # one sample so no uncertainty computed

    lenp = len(list(
        product(
            combinations(camin.range_a, dim_nmgkp),
            combinations(camin.range_i, dim_nmgkp))))

    assert alphas_min.shape[0] == lenp
    assert sigalphas_min.shape[0] == lenp

    assert np.isclose(camin.alphaNP_GKP(), alphas_min[0], atol=0, rtol=3)

    vd, sig_vd, v1, sig_v1, xinds = sample_gkp_parts(
        elem=camin,
        nsamples=1000,
        dim=dim_nmgkp,
        detstr="nmgkp")

    alphas, sigalphas = assemble_gkp_combinations(
        elem=camin,
        meanvoldat=vd,
        sigvoldat=sig_vd,
        meanvol1=v1,
        sigvol1=sig_v1,
        xindlist=xinds,
        dim=3,
        detstr="nmgkp")

    UB = alphas + 2 * sigalphas
    LB = alphas - 2 * sigalphas

    assert np.all(LB < 0)
    assert np.all(UB > 0)

    assert np.isclose(camin.alphaNP_GKP(), alphas[0], atol=0, rtol=4)


def test_proj_combinations():

    camin = Elem("Camin")
    dim_proj = 3

    (
        meanfrac, sigfrac, xindlist_proj
    ) = sample_proj_parts(camin, 1, dim_proj)
    # one sample so no uncertainty computed

    alphas_proj, sigalphas_proj = assemble_proj_combinations(
        camin, meanfrac, sigfrac, xindlist_proj)

    lenp_proj = len(list(
        product(
            combinations(camin.range_a, dim_proj),
            combinations(camin.range_i, 2))))
    # one sample so no uncertainty computed

    assert alphas_proj.shape[0] == lenp_proj
    assert sigalphas_proj.shape[0] == lenp_proj

    assert np.isclose(camin.alphaNP_proj(), alphas_proj[0], atol=0, rtol=1e-28)
    print()
    print("test_proj_combinations")
    (
        meanfrac, sigfrac, xindlist_proj
    ) = sample_proj_parts(camin, 1000, dim_proj)

    print("sigfrac", sigfrac)

    alphas_proj, sigalphas_proj = assemble_proj_combinations(
        camin, meanfrac, sigfrac, xindlist_proj)

    print("sigalphas_proj", sigalphas_proj)
    print()

    lenp_proj = len(list(
        product(
            combinations(camin.range_a, dim_proj),
            combinations(camin.range_i, 2))))

    assert alphas_proj.shape[0] == lenp_proj
    assert sigalphas_proj.shape[0] == lenp_proj

    # print("camin.alphaNP_proj()", camin.alphaNP_proj())
    # print("alphas_proj[0]      ", alphas_proj[0])

    assert np.isclose(camin.alphaNP_proj(), alphas_proj[0], atol=0, rtol=1)

    ca24 = Elem("Ca_WT_Aarhus_2024")
    dim_proj = 4

    alphaca = ca24.alphaNP_proj(ainds=[0, 1, 2, 3], iinds=[0, 1])

    (
        meanfrac, sigfrac, xindlist_proj
    ) = sample_proj_parts(ca24, 1000, dim_proj)

    alphas_proj, sigalphas_proj = assemble_proj_combinations(
        ca24, meanfrac, sigfrac, xindlist_proj)

    abs_alphaNP_proj_cv_Mathematica = 2.45796e-11
    abs_alphaNP_proj_UB_Mathematica = 1.35176e-10

    sig_alphaNP_proj_Mathematica = (
        abs_alphaNP_proj_UB_Mathematica - abs_alphaNP_proj_cv_Mathematica) / 2

    assert np.isclose(sig_alphaNP_proj_Mathematica, sigalphas_proj[0],
        atol=0, rtol=1e-2)

    assert np.isclose(np.abs(alphas_proj[0]), alphaca, atol=0, rtol=10)
    assert np.isclose(np.abs(alphas_proj[0]), abs_alphaNP_proj_cv_Mathematica,
        atol=0, rtol=10)


if __name__ == "__main__":
    test_linfit()
    test_GKP_combinations()
    test_NMGKP_combinations()
    test_proj_combinations()
