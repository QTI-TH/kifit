import numpy as np
import logging
import os

from kifit.build import Elem
from kifit.fitools import (
    generate_elemsamples, generate_alphaNP_samples,
    get_llist_elemsamples, get_delchisq, get_delchisq_crit, get_confint)
from kifit.detools import (
    sample_gkp_parts, assemble_gkp_combinations,
    sample_proj_parts, assemble_proj_combinations
)

np.random.seed(1)
fsize = 12
axislabelsize = 15

plotfolder = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "test_output"))
if not os.path.exists(plotfolder):
    logging.info("Creating file at ", plotfolder)
    os.mkdir(plotfolder)

def get_det_vals(elem, nsamples, dim, method_tag):

    (
        meanvd, sigvd, meanv1, sigv1, xindlist
    ) = sample_gkp_parts(elem, nsamples, dim, method_tag)

    alphas, sigalphas = assemble_gkp_combinations(
        elem, meanvd, sigvd, meanv1, sigv1, xindlist, dim, method_tag)

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


def test_d_swap_varying_inputparams():

    camin = Elem('Camin')
    camin_swap = Elem('Camin_swap')

    swapped_swap_dmat = swap_dmat(camin_swap.dmat(True))
    assert np.allclose(swapped_swap_dmat[0], camin.dmat(True)[0], atol=0,
                       rtol=1e-4)
    assert np.allclose(swapped_swap_dmat[1], camin.dmat(True)[1], atol=0,
                       rtol=1e-5)
    assert np.allclose(swapped_swap_dmat[2], camin.dmat(True)[2], atol=0,
                       rtol=1e-4)

    swapped_swap_dmat = swap_dmat(camin_swap.dmat(False))
    assert np.allclose(swapped_swap_dmat[0], camin.dmat(False)[0], atol=0,
                       rtol=1e-4)
    assert np.allclose(swapped_swap_dmat[1], camin.dmat(False)[1], atol=0,
                       rtol=1e-5)
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
                           rtol=1e-4)


        assert np.isclose(camin.dnorm, camin_swap.dnorm, atol=0, rtol=1e-100)

        camin_absdsamples_alpha0.append(camin.absd(True))
        camin_swap_absdsamples_alpha0.append(camin_swap.absd(True))

    camin_covmat_alpha0 = np.cov(np.array(camin_absdsamples_alpha0),
                                 rowvar=False)
    camin_swap_covmat_alpha0 = np.cov(np.array(camin_swap_absdsamples_alpha0),
                                      rowvar=False)

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
                           rtol=1e-7)
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

        camin_absdsamples_alpha1.append(camin.absd(True))
        camin_swap_absdsamples_alpha1.append(camin_swap.absd(True))

    camin_covmat_alpha1 = np.cov(np.array(camin_absdsamples_alpha1),
                                 rowvar=False)
    camin_swap_covmat_alpha1 = np.cov(np.array(camin_swap_absdsamples_alpha1),
                                      rowvar=False)

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
                           rtol=1e-9)

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

    assert np.allclose(camin_covmat_alpha2, camin_swap_covmat_alpha2,
        atol=0, rtol=1e-8)

    camin_ll_alpha2 = get_llist_elemsamples(camin_absdsamples_alpha2)
    camin_swap_ll_alpha2 = get_llist_elemsamples(camin_swap_absdsamples_alpha2)

    assert np.allclose(camin_ll_alpha2, camin_swap_ll_alpha2, atol=0, rtol=1e-8)


def Elem_swap_loop(
        elem,
        inputparamsamples,
        fitparamsamples,
        alphasamples,
        min_percentile,
        elem_swap=None,
        only_inputparams=False,
        only_fitparams=False,
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
            k1 = fitparams[0]
            ph1 = fitparams[1]

            if np.isclose(ph1, 0., atol=0, rtol=1e-17):
                raise Exception("ph1 is too close to zero")

            k1swap = k1
            ph1swap = np.arctan(1 / np.tan(ph1))

            if not only_inputparams:
                elem._update_fit_params([k1, ph1, alphaNP])
                if elem_swap is not None:
                    elem_swap._update_fit_params([k1swap, ph1swap, alphaNP])
            else:
                elem.alphaNP = alphaNP
                if elem_swap is not None:
                    elem_swap.alphaNP = alphaNP

            # update inputparams
            if not only_fitparams:
                inputparams = inputparamsamples[i]
                elem._update_elem_params(inputparams)
            elem_absdsamples.append(elem.absd(symm))

            if elem_swap is not None:
                if not only_fitparams:
                    inputparamswap = swap_inputparams(inputparams,
                                                      elem.nisotopepairs,
                                                      elem.ntransitions)
                    elem_swap._update_elem_params(inputparamswap)
                elem_swap_absdsamples.append(elem_swap.absd(symm))

        elem_ll = get_llist_elemsamples(elem_absdsamples, lam=lam)
        elem_ll_elemfit.append(np.percentile(elem_ll, min_percentile))

        if elem_swap is not None:
            # assert np.allclose(elem_absdsamples, elem_swap_absdsamples,
            #                    atol=0, rtol=1)  # rtol=1e-1)
            elem_covmat = (np.cov(np.array(elem_absdsamples), rowvar=False)
                           + 1e-17 * np.eye(elem.nisotopepairs))
            elem_swap_covmat = (np.cov(np.array(elem_swap_absdsamples),
                                rowvar=False)
                                + 1e-17 * np.eye(elem.nisotopepairs))
            # assert np.allclose(elem_covmat, elem_swap_covmat, atol=0, rtol=1e-2)
            #rtol=1e-3)

            elem_swap_ll = get_llist_elemsamples(elem_swap_absdsamples, lam=lam)
            # assert np.allclose(elem_ll, elem_swap_ll, atol=0, rtol=1e-1)  # rtol=1e-2)
            #
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

    camin_delchisq_caminfit, camin_swap_delchisq_caminfit = Elem_swap_loop(
        camin,
        inputparamsamples,
        fitparamsamples,
        alphasamples,
        min_percentile,
        elem_swap=camin_swap,
        only_inputparams=only_inputparams,
        symm=symm)
    camin_confint = get_confint(alphasamples, camin_delchisq_caminfit, delchisqcrit)

    camin_swap_delchisq_caminswapfit, camin_delchisq_caminswapfit = Elem_swap_loop(
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
               label="Ca 01, Ca 01 samples", s=15)
    ax.scatter(alphasamples, camin_swap_delchisq_caminfit, color='b',
               label="Ca 10, Ca 01 samples", s=5)
    ax.scatter(alphasamples, camin_delchisq_caminswapfit, color='orange',
               label="Ca 01, Ca 10 samples", s=15)
    ax.scatter(alphasamples, camin_swap_delchisq_caminswapfit, color='c',
               label="Ca 10, Ca 10 samples", s=5)

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
              + "Ca 01:\n"
              + r"$\langle\frac{\alpha_{\mathrm{NP}}}{\alpha_{\mathrm{EM}}}\rangle =$"
              + f"{camin_alpha_det[0]:.1e}\n"
              + r"$\frac{\alpha_{\mathrm{NP}}}{\alpha_{\mathrm{EM}}}\in$ ["
              + f"{camin_LB_det[0]:.1e}, {camin_UB_det[0]:.1e}"
              + "]\n"
              + "Ca 10:\n"
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
        plotfolder,
        f"mc_output_Camin_01_10_{symmstr}x{camin.x}_{varstr}.pdf")
    plt.savefig(plotpath, dpi=1000)

    # if not only_inputparams:
    ax.set_ylim(0, 20)
    ax.set_xlim(-1e-10, 1e-10)

    plotpath = os.path.join(
        plotfolder,
        f"mc_output_Camin_10_01_{symmstr}x{camin.x}_{varstr}_zoom.pdf")
    plt.savefig(plotpath, dpi=1000)


def test_swap():
    swap_varying_elemparams(only_inputparams=False, symm=False)
    swap_varying_elemparams(only_inputparams=True, symm=False)
    swap_varying_elemparams(only_inputparams=False, symm=True)
    swap_varying_elemparams(only_inputparams=True, symm=True)


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

    plt.legend(fontsize=fsize)

    ax.set_ylim(0, 1)
    ax.set_xlim(-.5e-10, .5e-10)

    ax.tick_params(axis="both", which="major", labelsize=fsize)
    ax.xaxis.get_offset_text().set_fontsize(fsize)

    plotpath = os.path.join(plotfolder,
                            f"mc_output_Camin_lam_x{camin.x}.pdf")
    plt.savefig(plotpath, dpi=1000)

    ax.set_ylim(0, 1)
    ax.set_xlim(-.5e-10, .5e-10)
    ax.tick_params(axis="both", which="major", labelsize=fsize)
    ax.xaxis.get_offset_text().set_fontsize(fsize)

    plotpath = os.path.join(plotfolder,
                            f"mc_output_Camin_lam_x{camin.x}_zoom.pdf")
    plt.savefig(plotpath, dpi=1000)


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

    delchisqs_elemvar = Elem_swap_loop(camin,
                                       inputparamsamples,
                                       fitparamsamples,
                                       alphasamples,
                                       min_percentile=min_percentile,
                                       only_inputparams=True,
                                       symm=symm)
    confint_elemvar = get_confint(alphasamples, delchisqs_elemvar, delchisqcrit)

    delchisqs_elemfitvar = Elem_swap_loop(camin,
                                          inputparamsamples,
                                          fitparamsamples,
                                          alphasamples,
                                          min_percentile=min_percentile,
                                          only_inputparams=False,
                                          symm=symm)
    confint_elemfitvar = get_confint(alphasamples, delchisqs_elemfitvar, delchisqcrit)

    delchisqs_fitvar = Elem_swap_loop(camin,
                                      inputparamsamples,
                                      fitparamsamples,
                                      alphasamples,
                                      min_percentile=min_percentile,
                                      only_fitparams=True,
                                      lam=1e-20,
                                      symm=symm)
    confint_fitvar = get_confint(alphasamples, delchisqs_fitvar, delchisqcrit)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(alphasamples, delchisqs_elemvar,
               label=(r"varying input parameters only: $\alpha_{\mathrm{NP}}\in$"
                      + f"[{confint_elemvar[0]:.1e},{confint_elemvar[1]:.1e}]"),
               s=5, color='b')
    ax.scatter(alphasamples, delchisqs_elemfitvar,
               label=(r"varying input params. & $K, \phi$: "
                      + r"$\alpha_{\mathrm{NP}}\in$"
                      +
                      f"[{confint_elemfitvar[0]:.1e},{confint_elemfitvar[1]:.1e}]"),
               s=5, color='purple')
    ax.scatter(alphasamples, delchisqs_fitvar,
               label=(r"varying fit params. $K, \phi$ only:"
                      + r"$\alpha_{\mathrm{NP}}\in$"
                      + f"[{confint_fitvar[0]:.1e},{confint_fitvar[1]:.1e}]"),
               s=5, color='orange')

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
    plotpath = os.path.join(plotfolder,
                            f"mc_output_elemvar_vs_elemfitvar_x{camin.x}"
                            + ("_symm" if symm else "")
                            + ".pdf")
    plt.savefig(plotpath, dpi=1000)

def test_elemvar_vs_elemfitvar():
    plot_elemvar_vs_elemfitvar(symm=True)
    plot_elemvar_vs_elemfitvar(symm=False)

if __name__ == "__main__":
    test_d_swap_varying_inputparams()
    test_swap()
    test_lam()
    test_elemvar_vs_elemfitvar()
