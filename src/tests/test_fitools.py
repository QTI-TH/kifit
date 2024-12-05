import numpy as np
import os

from kifit.build import Elem
from kifit.fitools import (
    generate_elemsamples, generate_alphaNP_samples,
    get_llist_elemsamples, get_delchisq)
np.random.seed(1)


# next add test of swap camin_swap vs camin, ref trans 1

def test_d_swap_varying_inputparams():

    camin = Elem('Camin')
    camin_swap = Elem('Camin_swap')

    camin_kp1_orig = camin.kp1
    camin_ph1_orig = camin.ph1

    swap_dmat = (camin_swap.dmat).T
    swapped_swap_dmat = np.c_[swap_dmat[1], swap_dmat[0]]

    assert np.allclose(swapped_swap_dmat[0], camin.dmat[0], atol=0, rtol=1e-4)
    assert np.allclose(swapped_swap_dmat[1], camin.dmat[1], atol=0, rtol=1e-5)
    assert np.allclose(swapped_swap_dmat[2], camin.dmat[2], atol=0, rtol=1e-4)

    assert np.isclose(camin.dnorm, camin_swap.dnorm, atol=0, rtol=1e-100)

    inputparamsamples, _ = generate_elemsamples(camin, 5)

    # alphaNP = 0
    ###########################################################################

    camin_absdsamples_alpha0 = []
    camin_swap_absdsamples_alpha0 = []

    for i, inputparams in enumerate(inputparamsamples):

        inputparamat = np.reshape(inputparams[2 * camin.nisotopepairs:],
            (camin.nisotopepairs, camin.ntransitions)).T

        inputparamat_swap = np.c_[inputparamat[1], inputparamat[0]]

        inputparamswap = np.concatenate((inputparams[: 2 * camin.nisotopepairs],
            inputparamat_swap), axis=None)

        camin._update_elem_params(inputparams)
        camin_swap._update_elem_params(inputparamswap)

        swap_dmat = (camin_swap.dmat).T
        swapped_swap_dmat = np.c_[swap_dmat[1], swap_dmat[0]]

        assert np.allclose(swapped_swap_dmat[0], camin.dmat[0], atol=0, rtol=1e-3)
        assert np.allclose(swapped_swap_dmat[1], camin.dmat[1], atol=0, rtol=1e-3)
        assert np.allclose(swapped_swap_dmat[2], camin.dmat[2], atol=0, rtol=1e-2)

        assert np.isclose(camin.dnorm, camin_swap.dnorm, atol=0, rtol=1e-100)

        camin_absdsamples_alpha0.append(camin.absd)
        camin_swap_absdsamples_alpha0.append(camin_swap.absd)

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

        inputparamat = np.reshape(inputparams[2 * camin.nisotopepairs:],
            (camin.nisotopepairs, camin.ntransitions)).T

        inputparamat_swap = np.c_[inputparamat[1], inputparamat[0]]

        inputparamswap = np.concatenate((inputparams[: 2 * camin.nisotopepairs],
            inputparamat_swap), axis=None)

        camin._update_elem_params(inputparams)
        camin_swap._update_elem_params(inputparamswap)

        swap_dmat = (camin_swap.dmat).T
        swapped_swap_dmat = np.c_[swap_dmat[1], swap_dmat[0]]

        assert np.allclose(swapped_swap_dmat[0], camin.dmat[0], atol=0, rtol=1e-7)
        assert np.allclose(swapped_swap_dmat[1], camin.dmat[1], atol=0, rtol=1e-7)
        assert np.allclose(swapped_swap_dmat[2], camin.dmat[2], atol=0, rtol=1e-8)

        assert np.isclose(camin.dnorm, camin_swap.dnorm, atol=0, rtol=1e-100)

        camin_absdsamples_alpha1.append(camin.absd)
        camin_swap_absdsamples_alpha1.append(camin_swap.absd)

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

        inputparamat = np.reshape(inputparams[2 * camin.nisotopepairs:],
            (camin.nisotopepairs, camin.ntransitions)).T

        inputparamat_swap = np.c_[inputparamat[1], inputparamat[0]]

        inputparamswap = np.concatenate((inputparams[: 2 * camin.nisotopepairs],
            inputparamat_swap), axis=None)

        camin._update_elem_params(inputparams)
        camin_swap._update_elem_params(inputparamswap)

        swap_dmat = (camin_swap.dmat).T
        swapped_swap_dmat = np.c_[swap_dmat[1], swap_dmat[0]]

        assert np.allclose(swapped_swap_dmat[0], camin.dmat[0], atol=0, rtol=1e-8)
        assert np.allclose(swapped_swap_dmat[1], camin.dmat[1], atol=0, rtol=1e-8)
        assert np.allclose(swapped_swap_dmat[2], camin.dmat[2], atol=0, rtol=1e-9)

        assert np.isclose(camin.dnorm, camin_swap.dnorm, atol=0, rtol=1e-100)

        camin_absdsamples_alpha2.append(camin.absd)
        camin_swap_absdsamples_alpha2.append(camin_swap.absd)

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
        elem_swap,
        inputparamsamples,
        fitparamsamples,
        alphasamples,
        min_percentile,
        only_inputparams = False
        ):

    elem_ll_elemfit = []
    elem_swap_ll_elemfit = []

    for alphaNP in alphasamples:

        elem_absdsamples = []
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
                elem_swap._update_fit_params([kp1swap, ph1swap, alphaNP])
            else:
                elem.alphaNP = alphaNP
                elem_swap.alphaNP = alphaNP

            # update inputparams
            inputparams = inputparamsamples[i]

            inputparamat = np.reshape(inputparams[2 * elem.nisotopepairs:],
                (elem.nisotopepairs, elem.ntransitions)).T

            inputparamat_swap = np.c_[inputparamat[1], inputparamat[0]]

            inputparamswap = np.concatenate((inputparams[: 2 * elem.nisotopepairs],
                inputparamat_swap), axis=None)

            elem._update_elem_params(inputparams)
            elem_swap._update_elem_params(inputparamswap)

            # compute abs(d)
            elem_absdsamples.append(elem.absd)
            elem_swap_absdsamples.append(elem_swap.absd)

        assert np.allclose(elem_absdsamples, elem_swap_absdsamples, atol=0,
                           rtol=1e-1)

        elem_covmat = (np.cov(np.array(elem_absdsamples), rowvar=False)
                        + 1e-17 * np.eye(elem.nisotopepairs))
        elem_swap_covmat = (np.cov(np.array(elem_swap_absdsamples),
                                    rowvar=False)
                             + 1e-17 * np.eye(elem.nisotopepairs))

        assert np.allclose(elem_covmat, elem_swap_covmat, atol=0, rtol=1e-3)

        elem_ll = get_llist_elemsamples(elem_absdsamples, lam=1e-15)
        elem_swap_ll = get_llist_elemsamples(elem_swap_absdsamples, lam=1e-15)

        assert np.allclose(elem_ll, elem_swap_ll, atol=0, rtol=1e-2)

        elem_ll_elemfit.append(np.percentile(elem_ll, min_percentile))
        elem_swap_ll_elemfit.append(np.percentile(elem_swap_ll,
                                                      min_percentile))

    elem_delchisq_elemfit = get_delchisq(elem_ll_elemfit, minll=None)
    elem_swap_delchisq_elemfit = get_delchisq(elem_swap_ll_elemfit,
                                                minll=None)

    return elem_delchisq_elemfit, elem_swap_delchisq_elemfit


def swap_varying_elemparams(only_inputparams = False):

    # varying both input parameters and fit parameters

    camin = Elem('Camin')
    camin_swap = Elem('Camin_swap')

    nalphasamples = 100
    nelemsamples = 100
    min_percentile = 5

    inputparamsamples, fitparamsamples = generate_elemsamples(camin, nelemsamples)
    inputparamsamples_swap, fitparamsamples_swap = generate_elemsamples(
            camin, nelemsamples)

    if only_inputparams:
        camin.set_alphaNP_init(0, 1e-2)

    else:
        camin.set_alphaNP_init(0, 5e-11)

    alphasamples = generate_alphaNP_samples(
        camin,
        nalphasamples,
        search_mode="normal")

    camin_delchisq_caminfit, camin_swap_delchisq_caminfit = Elem_swap_loop(
        camin,
        camin_swap,
        inputparamsamples,
        fitparamsamples,
        alphasamples,
        min_percentile,
        only_inputparams = only_inputparams)

    camin_swap_delchisq_caminswapfit, camin_delchisq_caminswapfit = Elem_swap_loop(
        camin_swap,
        camin,
        inputparamsamples_swap,
        fitparamsamples_swap,
        alphasamples,
        min_percentile,
        only_inputparams = only_inputparams)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.scatter(alphasamples, camin_delchisq_caminfit, alpha=0.5, color='r',
               label="Camin Caminfit", s=15)
    ax.scatter(alphasamples, camin_swap_delchisq_caminfit, alpha=0.5,
               color='b', label="Camin_swap Caminfit", s=5)
    ax.scatter(alphasamples, camin_delchisq_caminswapfit, alpha=0.5, color='orange',
               label="Camin Caminswapfit", s=15)
    ax.scatter(alphasamples, camin_swap_delchisq_caminswapfit, alpha=0.5,
               color='c', label="Camin_swap Caminswapfit", s=5)

    ax.set_xlabel(r"$\alpha_{\mathrm{NP}} / \alpha_{\mathrm{EM}}$")
    ax.set_ylabel(r"$\Delta \chi^2$")

    plt.title(
            f"Camin vs. Camin_swap at x={camin.x}, {len(alphasamples)}"
            + r" $\alpha_{\mathrm{NP}}$ samples")
    plt.legend()

    if only_inputparams:
        varstr = "inputparamvar"
    else:
        varstr = "elemparamvar"

    plotpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"mc_output_Camin_Camin_swap_x{camin.x}_{varstr}.pdf")
    plt.savefig(plotpath, dpi=1000)

    if not only_inputparams:
        ax.set_ylim(0, 10)
        ax.set_xlim(-1e-10, 1e-10)

    plotpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"mc_output_Camin_Camin_swap_x{camin.x}_{varstr}_zoom.pdf")
    plt.savefig(plotpath, dpi=1000)


def test_swap():
    swap_varying_elemparams(only_inputparams = False)
    swap_varying_elemparams(only_inputparams = True)


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

    camin_delchisq = []

    for lamval in lamvals:

        camin_ll_lam = []

        for alphaNP in alphasamples:

            camin_absdsamples = []

            for i, fitparams in enumerate(fitparamsamples):

                fitparams.append(alphaNP)

                if np.isclose(fitparams[1], 0., atol=0, rtol=1e-17):
                    raise Exception("ph1 is too close to zero")

                camin._update_fit_params(fitparams)
                camin._update_elem_params(inputparamsamples[i])

                camin_absdsamples.append(camin.absd)

            camin_ll = get_llist_elemsamples(camin_absdsamples, lam=lamval)
            camin_ll_lam.append(np.percentile(camin_ll, min_percentile))

        camin_delchisq.append(get_delchisq(camin_ll_lam, minll=None))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    for l, lam in enumerate(lamvals):
        ax.scatter(alphasamples, camin_delchisq[l], label=f"lam={lam}", s=5)

    ax.set_xlabel(r"$\alpha_{\mathrm{NP}} / \alpha_{\mathrm{EM}}$")
    ax.set_ylabel(r"$\Delta \chi^2$")

    plt.title(
            f"Camin at x={camin.x}, {len(alphasamples)}"
            + r" $\alpha_{\mathrm{NP}}$ samples")
    plt.legend()

    plotpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"mc_output_Camin_lam_x{camin.x}.pdf")
    plt.savefig(plotpath, dpi=1000)

    ax.set_ylim(0, 1)
    ax.set_xlim(-.5e-10, .5e-10)

    plotpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"mc_output_Camin_lam_x{camin.x}_zoom.pdf")
    plt.savefig(plotpath, dpi=1000)


if __name__ == "__main__":
    test_swap()
    test_lam()
