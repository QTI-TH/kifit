import numpy as np

from kifit.build import Elem
from kifit.fitvsdetools import (
        estimate_covnutil, grad_alphaNP3, Pperp, Pparallel, eFvec,
        alphaNP_GKP3_nutil, SigmalphaNP)

np.random.seed(1)


def test_SigmalphaNP():

    camin = Elem("Camin")
    mean_nutil = camin.nutil

    covca = estimate_covnutil(camin, 1000)
    assert np.allclose(covca, covca.T, atol=0, rtol=1e-25)

    assert np.allclose(camin.eF, eFvec(camin))

    Para = Pparallel(camin)
    print("Ppara", Para)
    assert np.allclose(Para @ Para, Para, atol=0, rtol=1e-5)

    Perp = Pperp(camin)
    assert np.allclose(Perp @ Perp, Perp, atol=0, rtol=1e-5)

    print("Perp @ Para", Perp @ Para)
    print("Perp @ Para", (Perp @ Para).shape)
    print("Perp       ", Perp.shape)

    assert np.allclose(Perp @ Para, np.zeros_like(Perp), atol=1e-15)

    grad_alphaNP = grad_alphaNP3(camin)

    # test convergence h->0 in grad alphaNP
    print("h=1000", grad_alphaNP3(camin, h=1000))
    print("h=100", grad_alphaNP3(camin, h=100))
    print("h=10", grad_alphaNP3(camin, h=10))
    print("h=1", grad_alphaNP3(camin, h=1))
    print("h=1e-1", grad_alphaNP3(camin, h=1e-1))
    print("h=1e-2", grad_alphaNP3(camin, h=1e-2))
    print("h=1e-3", grad_alphaNP3(camin, h=1e-3))
    print("h=1e-4", grad_alphaNP3(camin, h=1e-4))
    print("h=1e-5", grad_alphaNP3(camin, h=1e-5))
    print("h=1e-6", grad_alphaNP3(camin, h=1e-6))

    assert np.allclose(
            grad_alphaNP3(camin, h=1000), grad_alphaNP3(camin, h=100),
            atol=1e-20, rtol=0)
    assert np.allclose(
            grad_alphaNP3(camin, h=1000), grad_alphaNP3(camin, h=100),
            atol=0, rtol=1e-7)

    assert np.allclose(
            grad_alphaNP3(camin, h=100), grad_alphaNP3(camin, h=10),
            atol=1e-20, rtol=0)
    assert np.allclose(
            grad_alphaNP3(camin, h=100), grad_alphaNP3(camin, h=10),
            atol=0, rtol=1e-7)

    assert np.allclose(
            grad_alphaNP3(camin, h=10), grad_alphaNP3(camin, h=1),
            atol=1e-20, rtol=0)
    assert np.allclose(
            grad_alphaNP3(camin, h=10), grad_alphaNP3(camin, h=1),
            atol=0, rtol=1e-6)

    assert np.allclose(
            grad_alphaNP3(camin, h=1), grad_alphaNP3(camin, h=1e-1),
            atol=1e-19, rtol=0)
    assert np.allclose(
            grad_alphaNP3(camin, h=1), grad_alphaNP3(camin, h=1e-1),
            atol=0, rtol=1e-5)

    assert np.allclose(
            grad_alphaNP3(camin, h=1e-1), grad_alphaNP3(camin, h=1e-2),
            atol=1e-18, rtol=0)
    assert np.allclose(
            grad_alphaNP3(camin, h=1e-1), grad_alphaNP3(camin, h=1e-2),
            atol=0, rtol=1e-5)

    assert np.allclose(
            grad_alphaNP3(camin, h=1e-2), grad_alphaNP3(camin, h=1e-3),
            atol=1e-16, rtol=0)
    assert np.allclose(
            grad_alphaNP3(camin, h=1e-2), grad_alphaNP3(camin, h=1e-3),
            atol=0, rtol=1e-3)

    assert np.allclose(
            grad_alphaNP3(camin, h=1e-3), grad_alphaNP3(camin, h=1e-4),
            atol=1e-16, rtol=0)
    assert np.allclose(
            grad_alphaNP3(camin, h=1e-3), grad_alphaNP3(camin, h=1e-4),
            atol=0, rtol=1e-2)

    assert np.allclose(
            grad_alphaNP3(camin, h=1e-4), grad_alphaNP3(camin, h=1e-5),
            atol=1e-15, rtol=0)
    assert np.allclose(
            grad_alphaNP3(camin, h=1e-4), grad_alphaNP3(camin, h=1e-5),
            atol=0, rtol=1e-1)

    assert np.allclose(
            grad_alphaNP3(camin, h=1e-5), grad_alphaNP3(camin, h=1e-6),
            atol=1e-14, rtol=0)
    assert np.allclose(
            grad_alphaNP3(camin, h=1e-5), grad_alphaNP3(camin, h=1e-6),
            atol=0, rtol=1e-1)

    assert np.allclose(
            grad_alphaNP3(camin, h=1e-6), grad_alphaNP3(camin, h=1e-7),
            atol=1e-13, rtol=0)

    # grad_alphaNP3(camin, h=1e-7)) = [0. 0. 0. 0. 0. 0.]
    # grad_alphaNP3(camin, h=1e-8)) = [0. 0. 0. 0. 0. 0.]



    print("camin.alphaNP_GKP()", camin.alphaNP_GKP())
    assert np.isclose(
            camin.alphaNP_GKP(),
            alphaNP_GKP3_nutil(mean_nutil, camin.Xvec, camin.gammatilvec),
            atol=0, rtol=1e-25)


    Sigmalpha, Sigmalpha_perp, Sigmalpha_parallel = SigmalphaNP(camin,
                                                                nsamples=1000)
    assert np.isclose(Sigmalpha, Sigmalpha_perp + Sigmalpha_parallel,
                      atol=0, rtol=1e-2)


if __name__ == "__main__":
    test_SigmalphaNP()
