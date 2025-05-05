import numpy as np
from itertools import product, combinations
import os

from kifit.build import Elem
from kifit.detools import (
        generate_alphaNP_dets, sample_gkp_combinations, sample_proj_combinations)
from kifit.plot import (gkp_colour, nmgkp_colour, proj_colour)

np.random.seed(1)
fsize = 12
axislabelsize = 15

plotfolder = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "test_output"))
if not os.path.exists(plotfolder):
    logging.info("Creating file at ", plotfolder)
    os.mkdir(plotfolder)


def test_GKP_combinations():

    camin = Elem("Camin")
    dim_gkp = 3

    # one sample so no uncertainty computed
    alphas_min, sigalphas_min, num_perm = generate_alphaNP_dets(
            elem=camin,
            nsamples=1,
            dim=dim_gkp,
            detstr="gkp")

    lenp_gkp = len(list(
        product(
            combinations(camin.range_a, dim_gkp),
            combinations(camin.range_i, dim_gkp - 1))))

    assert num_perm == lenp_gkp
    assert alphas_min.shape[0] == lenp_gkp
    assert sigalphas_min.shape[0] == lenp_gkp

    assert np.isclose(
            camin.alphaNP_GKP(ainds=[0, 1, 2], iinds=[0, 1]),
            alphas_min[0],
            atol=0, rtol=1e-13)

    assert np.isclose(alphas_min[0], camin.alphaNP_GKP_combinations(3)[0],
        atol=0, rtol=1e-25)

    # 1e4 sample MC
    alphas, sigalphas, num_perm = generate_alphaNP_dets(
            elem=camin,
            nsamples=10000,
            dim=dim_gkp,
            detstr="gkp")

    alphaNP_GKP_Camin_kifit = alphas[0]
    sigalphaNP_GKP_Camin_kifit = sigalphas[0]

    # also MC with 1e4 samples
    alphaNP_GKP_Camin_Mathematica_1e4 = 1.36441192908216e-9
    sigalphaNP_GKP_Camin_Mathematica_1e4 = 1.119920127478992e-9

    # 1e4 elemsamples from kifit evaluated in Mathematica
    alphaNP_GKP_Camin_kifit_Mathematica_1e4 = 1.3392415021e-9
    sigalphaNP_GKP_Camin_kifit_Mathematica_1e4 = 1.1070042299e-9

    # analytic, sigalpha via grad(alphaNP)_nutil
    alphaNP_GKP_Camin_Mathematica_analytic = 1.362700911574815e-9
    sigalphaNP_GKP_Camin_Mathematica_analytic = 1.119369647625706e-9

    # assert np.isclose(camin.alphaNP_GKP(), alphas[0], atol=0, rtol=2)
    assert np.isclose(camin.alphaNP_GKP(),
                      alphaNP_GKP_Camin_kifit,
                      atol=0, rtol=1e-13)

    assert np.isclose(alphaNP_GKP_Camin_Mathematica_1e4,
                      alphaNP_GKP_Camin_kifit,
                      atol=0, rtol=1e-2)

    assert np.isclose(alphaNP_GKP_Camin_Mathematica_1e4,
                      alphaNP_GKP_Camin_kifit_Mathematica_1e4,
                      atol=0, rtol=1e-1)

    assert np.isclose(alphaNP_GKP_Camin_Mathematica_analytic,
                      alphaNP_GKP_Camin_kifit,
                      atol=0, rtol=1e-8)

    assert np.isclose(sigalphaNP_GKP_Camin_Mathematica_1e4,
                      sigalphaNP_GKP_Camin_kifit,
                      atol=0, rtol=1e-1)

    assert np.isclose(sigalphaNP_GKP_Camin_Mathematica_1e4,
                      sigalphaNP_GKP_Camin_kifit_Mathematica_1e4,
                      atol=0, rtol=1e-1)

    assert np.isclose(sigalphaNP_GKP_Camin_Mathematica_analytic,
                      sigalphaNP_GKP_Camin_kifit,
                      atol=0, rtol=1e-1)


    yb1 = Elem("strongest_Yb_Kyoto_MIT_GSI_2022")

    alphaNPs_yb1, sig_alphaNPs_yb1, num_perm_yb1 = generate_alphaNP_dets(
            elem=yb1,
            nsamples=1000,
            dim=3,
            detstr="gkp")

    yb1_UB = 9.018604e-11

    assert np.isclose(np.abs(alphaNPs_yb1[0]) + 2 * sig_alphaNPs_yb1[0], yb1_UB,
        atol=0, rtol=1e-1)

    yb2 = Elem("weakest_Yb_Kyoto_MIT_GSI_2022")

    alphaNPs_yb2, sig_alphaNPs_yb2, num_perm_yb2 = generate_alphaNP_dets(
            elem=yb2,
            nsamples=1000,
            dim=3,
            detstr="gkp")

    yb2_UB = 4.423329e-9

    assert np.isclose(alphaNPs_yb2[0] + 2 * sig_alphaNPs_yb2[0], yb2_UB,
        atol=0, rtol=1)

    yb3 = Elem("strongest_Yb_Kyoto_MIT_GSI_PTB_2024")

    alphaNPs_yb3, sig_alphaNPs_yb3, num_perm_yb3 = generate_alphaNP_dets(
        elem=yb3,
        nsamples=1000,
        dim=3,
        detstr="gkp")

    yb3_UB = 1.24674e-10

    assert np.isclose(np.abs(alphaNPs_yb3[0]) + 2 * sig_alphaNPs_yb3[0], yb3_UB,
        atol=0, rtol=1)

    yb4 = Elem("weakest_Yb_Kyoto_MIT_GSI_PTB_2024")

    alphaNPs_yb4, sig_alphaNPs_yb4, num_perm_yb4 = generate_alphaNP_dets(
        elem=yb4,
        nsamples=1000,
        dim=3,
        detstr="gkp")

    yb4_UB = 9.9457e-9

    assert np.isclose(np.abs(alphaNPs_yb4[0]) + 2 * sig_alphaNPs_yb4[0], yb4_UB,
        atol=0, rtol=1)


def test_NMGKP_combinations():

    camin = Elem("Ca_testdata")
    dim_nmgkp = 3

    # one sample so no uncertainty computed
    alphas_min, sigalphas_min, num_perm = generate_alphaNP_dets(
        elem=camin,
        nsamples=1,
        dim=dim_nmgkp,
        detstr="nmgkp")
    # one sample so no uncertainty computed

    lenp = len(list(
        product(
            combinations(camin.range_a, dim_nmgkp),
            combinations(camin.range_i, dim_nmgkp))))

    assert alphas_min.shape[0] == lenp
    assert sigalphas_min.shape[0] == lenp

    assert np.isclose(camin.alphaNP_NMGKP(), alphas_min[0], atol=0, rtol=1e-13)

    alphas, sigalphas, num_perm = generate_alphaNP_dets(
        elem=camin,
        nsamples=1000,
        dim=dim_nmgkp,
        detstr="nmgkp")

    UB = alphas + 2 * sigalphas
    LB = alphas - 2 * sigalphas

    assert np.all(LB < 0)
    assert np.all(UB > 0)

    assert np.isclose(camin.alphaNP_NMGKP(), alphas[0], atol=0, rtol=10)

def test_proj_combinations():

    camin = Elem("Camin")
    dim_proj = 3

    # one sample so no uncertainty computed
    alphas_proj, sigalphas_proj, num_perm = generate_alphaNP_dets(
            elem=camin,
            nsamples=1,
            dim=dim_proj,
            detstr="proj")

    lenp_proj = len(list(
        product(
            combinations(camin.range_a, dim_proj),
            combinations(camin.range_i, 2))))

    # one sample so no uncertainty computed
    assert alphas_proj.shape[0] == lenp_proj
    assert sigalphas_proj.shape[0] == lenp_proj

    assert np.isclose(camin.alphaNP_proj(), alphas_proj[0], atol=0, rtol=1e-28)

    alphas_proj, sigalphas_proj, num_perm = generate_alphaNP_dets(
            elem=camin,
            nsamples=4,
            # nsamples=1000,
            dim=dim_proj,
            detstr="proj")

    lenp_proj = len(list(
        product(
            combinations(camin.range_a, dim_proj),
            combinations(camin.range_i, 2))))

    assert alphas_proj.shape[0] == lenp_proj
    assert sigalphas_proj.shape[0] == lenp_proj

    assert np.isclose(camin.alphaNP_proj(), alphas_proj[0], atol=0, rtol=1)
    assert np.isclose(camin.alphaNP_GKP(), camin.alphaNP_proj(), atol=0,
                      rtol=1e-2)

    ca24 = Elem("Ca_WT_Aarhus_2024")
    dim_proj = 4

    alphaca = ca24.alphaNP_proj(ainds=[0, 1, 2, 3], iinds=[0, 1])

    alphas_proj, sigalphas_proj, num_perm = generate_alphaNP_dets(
            elem=ca24,
            nsamples=1000,
            dim=dim_proj,
            detstr="proj")

    abs_alphaNP_proj_cv_Mathematica = 2.45796e-11
    abs_alphaNP_proj_UB_Mathematica = 1.35176e-10

    sig_alphaNP_proj_Mathematica = (
        abs_alphaNP_proj_UB_Mathematica - abs_alphaNP_proj_cv_Mathematica) / 2

    assert np.isclose(sig_alphaNP_proj_Mathematica, sigalphas_proj[0],
        atol=0, rtol=1e-3)

    assert np.isclose(np.abs(alphas_proj[0]), alphaca, atol=0, rtol=10)
    assert np.isclose(np.abs(alphas_proj[0]), abs_alphaNP_proj_cv_Mathematica,
        atol=0, rtol=10)

    assert np.isclose(ca24.alphaNP_GKP(), ca24.alphaNP_proj(), atol=0, rtol=1e-4)

    alphaca1 = ca24.alphaNP_proj(ainds=[0, 1, 2], iinds=[0, 1])
    alphaca2 = ca24.alphaNP_proj(ainds=[0, 1, 3], iinds=[0, 1])
    alphaca3 = ca24.alphaNP_proj(ainds=[0, 2, 3], iinds=[0, 1])
    alphaca4 = ca24.alphaNP_proj(ainds=[1, 2, 3], iinds=[0, 1])

    alphas_proj_single = [alphaca1, alphaca2, alphaca3, alphaca4]

    alphas_proj_all, sigalphas_proj_all, num_perm_all = generate_alphaNP_dets(
            elem=ca24,
            nsamples=1000,
            dim=3,
            detstr="proj")

    assert np.allclose(alphas_proj_all, alphas_proj_single,
                       atol=0, rtol=1e-30)
    assert np.allclose(alphas_proj_single, alphas_proj_all,
                       atol=0, rtol=1e-30)

    catest = Elem("Ca_testdata")
    assert np.isclose(catest.alphaNP_GKP(), catest.alphaNP_proj(), atol=0,
                      rtol=1e-2)

    ca15 = Elem("Ca_PTB_2015")
    assert np.isclose(ca15.alphaNP_GKP(), ca15.alphaNP_proj(), atol=0,
                      rtol=1e-1)

    fullca = Elem("Ca_WT_Aarhus_PTB_2024")

    lama1 = fullca.alphaNP_proj(ainds=[0, 1, 2], iinds=[0, 1])
    lama2 = fullca.alphaNP_proj(ainds=[0, 1, 2], iinds=[0, 2])
    lama3 = fullca.alphaNP_proj(ainds=[0, 1, 2], iinds=[0, 3])
    lama4 = fullca.alphaNP_proj(ainds=[0, 1, 2], iinds=[1, 2])
    lama5 = fullca.alphaNP_proj(ainds=[0, 1, 2], iinds=[1, 3])
    lama6 = fullca.alphaNP_proj(ainds=[0, 1, 2], iinds=[2, 3])

    lamasingle = [lama1, lama2, lama3, lama4, lama5, lama6]

    lamall, siglamall, _ = generate_alphaNP_dets(
            elem=fullca,
            nsamples=1000,
            dim=3,
            detstr="proj")

    assert np.allclose(lamall, lamasingle,
                       atol=0, rtol=1e-30)
    assert np.allclose(lamasingle, lamall,
                       atol=0, rtol=1e-30)


def test_alphaNP_histogram(elemstr="Camin", nsamples=1000):

    elem = Elem(elemstr)
    dim_alg = 3

    alphasamples_gkp, _ = sample_gkp_combinations(
            elem=elem,
            nsamples=nsamples,
            dim=dim_alg,
            detstr="gkp")

    alphasamples_proj, _ = sample_proj_combinations(
            elem=elem,
            nsamples=nsamples,
            dim=dim_alg)

    Nperm = alphasamples_gkp.shape[1]

    if elem.ntransitions > 2:
        alphasamples_nmgkp, _ = sample_gkp_combinations(
                elem=elem,
                nsamples=nsamples,
                dim=dim_alg,
                detstr="nmgkp")
        Nperm_nmgkp = alphasamples_nmgkp.shape[1]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # remove outliers

    if elem.ntransitions > 2:
        allsamples = np.hstack((
            alphasamples_gkp.flatten(),
            alphasamples_nmgkp.flatten(),
            alphasamples_proj.flatten()
            ))
    else:
        allsamples = np.hstack((
            alphasamples_gkp.flatten(),
            alphasamples_proj.flatten()
            ))

    minsample = np.percentile(allsamples, 10)
    maxsample = np.percentile(allsamples, 90)

    filteredsamples = np.array([sample for sample in allsamples if
                                (sample > minsample and sample < maxsample)])

    bin_edges = np.histogram(filteredsamples, bins=int(nsamples/50))[1]

    for p in range(Nperm):
        if p == 0:
            gkp_label = f"dim-{dim_alg} GKP"
            proj_label = f"dim-{dim_alg} proj"
        else:
            gkp_label = ""
            proj_label = ""

        ax.hist(alphasamples_gkp[:, p], bins=bin_edges,
                histtype="step",
                color=gkp_colour, label=gkp_label)
        ax.hist(alphasamples_proj[:, p], bins=bin_edges,
                histtype="step", color=proj_colour, label=proj_label)

    if elem.ntransitions > 2:
        for p in range(Nperm_nmgkp):
            if p == 0:
                nmgkp_label = f"dim-{dim_alg} NMGKP"
            else:
                nmgkp_label = ""

            ax.hist(alphasamples_nmgkp[:, p], bins=bin_edges,
                    histtype="step", color=nmgkp_colour, label=nmgkp_label)


    ax.legend(fontsize=fsize)
    ax.set_xlabel(r"$\alpha_{\mathrm{NP}} / \alpha_{\mathrm{EM}}$",
                  fontsize=axislabelsize)
    ax.set_ylabel(f"Counts per {nsamples}")
    ax.set_title(elemstr)
    plotpath = os.path.join(
            plotfolder,
            f"alphaNP_histogram_{elemstr}_ns{nsamples}_gkproj.pdf")
    plt.savefig(plotpath)


if __name__ == "__main__":
    test_GKP_combinations()
    test_NMGKP_combinations()
    test_proj_combinations()
    test_alphaNP_histogram()
