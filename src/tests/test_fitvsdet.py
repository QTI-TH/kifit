import numpy as np

from kifit.build import Elem
from kifit.fitvsdetools import (
        Pperp, Pparallel, eFvec,
        covnutil_ai, V1_GKP_nutil, alphaNP_GKP_nutil, alphaNP_GKP_nutil_normed,
        grad_alphaNP_nutil,
        SigmalphaNP)
from Mathematica_crosschecks import (
        alphaNP_caPTB15, grad_alphaNP_caPTB15,
        alphaNP_camin, grad_alphaNP_camin,
        alphaNP_camin_swap, grad_alphaNP_camin_swap,
        alphaNP_ca24min, grad_alphaNP_ca24min)

np.random.seed(1)


def test_SigmalphaNP():

    camin = Elem("Camin")
    mean_nutil = camin.nutil

    covca = covnutil_ai(camin, ainds=[0, 1, 2], iinds=[0, 1], nsamples=1000)
    assert np.allclose(covca, covca.T, atol=0, rtol=1e-25)
    print("Camin covmat")
    print(covca)

    print("Camin alphaNP")
    print(camin.alphaNP_GKP())

    assert np.allclose(camin.eF, eFvec(camin))

    Para = Pparallel(camin)
    assert np.allclose(Para @ Para, Para, atol=0, rtol=1e-27)

    Perp = Pperp(camin)
    assert np.allclose(Perp @ Perp, Perp, atol=0, rtol=1e-11)

    assert np.allclose(Perp @ Para, np.zeros_like(Perp), atol=1e-17)

    assert np.isclose(camin.alphaNP_GKP(), alphaNP_camin, atol=0, rtol=1e-1)

    grad_alphaNP = grad_alphaNP_nutil(
            camin.nutil, camin.Xvec, camin.gammatilvec, np.ones(3))

    assert grad_alphaNP.shape == camin.nutil.flatten().shape

    # print("camin muvec", camin.muvec)
    # print("camin ma", camin.sig_m_a_in)
    # print("camin map", camin.sig_m_ap_in)
    print("grad_alphaNP", grad_alphaNP)
    print("mathematica ", grad_alphaNP_camin)
    # print("camin.alphaNP_GKP()", camin.alphaNP_GKP())
    # print("mathematica        ", alphaNP_camin)

    print("testi")
    print(camin.alphaNP_GKP())
    print(alphaNP_camin)
    print(alphaNP_GKP_nutil(mean_nutil, camin.Xvec, camin.gammatilvec))
    print(alphaNP_GKP_nutil_normed(mean_nutil, camin.Xvec, camin.gammatilvec))


    assert np.isclose(
            camin.alphaNP_GKP(),
            alphaNP_GKP_nutil(mean_nutil, camin.Xvec, camin.gammatilvec),
            atol=0, rtol=1e-10)


    Sigmalpha, Sigmalpha_perp, Sigmalpha_parallel = SigmalphaNP(camin,
                                                                nsamples=1000)
    # assert np.isclose(Sigmalpha, Sigmalpha_perp + Sigmalpha_parallel,
    #                   atol=0, rtol=1e-2)

    print("Sigmalpha_perp / Sigmalpha", Sigmalpha_perp / Sigmalpha)
    print("Sigmalpha_para / Sigmalpha", Sigmalpha_parallel / Sigmalpha)
    print("Sum / Sigmalpha", (Sigmalpha_parallel + Sigmalpha_perp) / Sigmalpha)


def test_SigmalphaNP_elems():

    ca15 = Elem("Ca_PTB_2015")
    Sa_ca15, Saperp_ca15, Sapara_ca15 = SigmalphaNP(ca15, nsamples=1000)

    ca20 = Elem("Ca_WT_Aarhus_2020")
    Sa_ca20, Saperp_ca20, Sapara_ca20 = SigmalphaNP(ca20, nsamples=1000)

    ca21 = Elem("Ca_WT_Aarhus_PTB_2020")
    Sa_ca21, Saperp_ca21, Sapara_ca21 = SigmalphaNP(ca21, nsamples=1000)

    ca24 = Elem("Ca_WT_Aarhus_2024")
    Sa_ca24, Saperp_ca24, Sapara_ca24 = SigmalphaNP(ca24, nsamples=1000)

    ca25 = Elem("Ca_WT_Aarhus_PTB_2024")
    Sa_ca25, Saperp_ca25, Sapara_ca25 = SigmalphaNP(ca25, nsamples=1000)

    camin = Elem("Camin")
    Sa_camin, Saperp_camin, Sapara_camin = SigmalphaNP(camin, nsamples=1000)

    print(                        "sig[alpha]   sig[alpha]_perp   sig[alpha]_para ")
    print("Ca_PTB_2015            ", [Sa_ca15, Saperp_ca15, Sapara_ca15])
    print("Ca_WT_Aarhus_2020      ", [Sa_ca20, Saperp_ca20, Sapara_ca20])
    print("Ca_WT_Aarhus_PTB_2020  ", [Sa_ca21, Saperp_ca21, Sapara_ca21])
    print("Ca_WT_Aarhus_2024      ", [Sa_ca24, Saperp_ca24, Sapara_ca24])
    print("Ca_WT_Aarhus_PTB_2024  ", [Sa_ca25, Saperp_ca25, Sapara_ca25])
    print("Camin                  ", [Sa_camin, Saperp_camin, Sapara_camin])



if __name__ == "__main__":
    test_SigmalphaNP()
    # test_SigmalphaNP_elems()
