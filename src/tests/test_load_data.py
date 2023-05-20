import numpy as np
from pprint import pprint
from qiss.loadelems import ElemData


def test_load_all():
    all_element_data = ElemData.load_all()

    assert np.allclose(
        np.nan_to_num(all_element_data['Ca'].nu).sum(axis=1),
        [8.44084572e+08, 1.68474592e+09, 7.75849948e+09, 3.38525526e+09],
        rtol=1e-8,
    )


def test_load_individual():
    ca = ElemData.get('Ca')
    pprint(ca)

    Ca = ElemData.get('Ca')
    assert (np.nan_to_num(ca.nu) == np.nan_to_num(Ca.nu)).all()
    print(Ca.nu)
    print(Ca.mnu)
    print(Ca.h_np_nucl)


def export_reduced_isotope_shifts():
    for elem in ['Ca']:
        elem_data = ElemData.get('Ca')
        np.savetxt('mnu_{}.dat'.format(elem), elem_data.mnu, delimiter=',')


if __name__ == "__main__":
    test_load_individual()
    export_reduced_isotope_shifts()
