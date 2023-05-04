from elements import ElemData
import numpy as np
from pprint import pprint



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
    #pprint(ca)
    print(Ca.nu)
    print(Ca.reduced_isotope_shifts)

    print(Ca.new_physics_term)



def export_new_physics_terms():
    for elem in ['Ca']:
        elem_data = ElemData.get('Ca')
        np.savetxt('new_physics_term_{}.dat'.format(elem), elem_data.new_physics_term, delimiter=',')


if __name__ == "__main__":
    export_new_physics_terms()
    test_load_individual()
