import os
import numpy as np
from functools import cache
from functools import cached_property
from qiss.user_elements import user_elems

_data_path = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../data'
))


class ElemData:
    # Load raw data from data folder
    # VALID_ELEM = ['Ca']
    VALID_ELEM = user_elems
    ELEM_FILES = ['nu', 'signu', 'Xcoeffs', 'isotopes']
    elem_init_atr = ['nu', 'sig_nu', 'Xcoeffs', 'isotope_data']

    CACHE = {}

    def __init__(self, element: str):
        assert element in self.VALID_ELEM, "Element {} not supported".format(element)
        print("Loading raw data")
        # load all data files associated to element
        self.id = element

        for (i, file_type) in enumerate(self.ELEM_FILES):
            assert len(self.ELEM_FILES) == len(self.elem_init_atr)
            file_name = file_type + self.id + '.dat'
            file_path = os.path.join(_data_path, self.id, file_name)
            assert os.path.exists(file_path), file_path
            self.__load(self.elem_init_atr[i], file_type, file_path)

    def __load(self, atr: str, file_type: str, file_path: str):
        print('loading attribute {} for element {} from {}'.format(atr, self.id, file_path))
        val = np.loadtxt(file_path)
        setattr(self, atr, val)

    def zeros(self):
        return np.zeros((self.m_isopair_nb, self.n_trans_nb))

    @classmethod
    def load_all(cls):
        """
        loads all elements and returns result as dict
        """
        return {u: cls(u) for u in cls.VALID_ELEM}

    def __repr__(self):
        return self.id + '[' + ','.join(self.elem_init_atr) + ']'

    @classmethod
    @cache
    def get(cls, elem: str):
        return cls(elem)

    @cached_property
    def a_isotope_nb(self):
        """
        Returns isotope numbers A

        """
        return self.isotope_data[0]

    @cached_property
    def ap_isotope_nb(self):
        """
        Returns isotope numbers A'

        """
        return self.isotope_data[3]

    @cached_property
    def m_a(self):
        """
        Returns masses of reference isotopes A

        """
        return self.isotope_data[2]

    @cached_property
    def m_ap(self):
        """
        Returns masses of isotopes A'

        """
        return self.isotope_data[4]

    @cached_property
    def sig_m_a(self):
        """
        Returns uncertainties on masses of reference isotopes A

        """
        return self.isotope_data[2]

    @cached_property
    def sig_m_ap(self):
        """
        Returns uncertainties on masses of isotopes A'

        """
        return self.isotope_data[5]

    @cached_property
    def m_isopair_nb(self):
        """
        Returns number of isotope pairs m

        """
        nisotopepairs = len(self.a_isotope_nb)

        assert nisotopepairs == len(self.ap_isotope_nb)
        assert nisotopepairs == len(self.m_a)
        assert nisotopepairs == len(self.m_ap)
        assert nisotopepairs == len(self.nu)
        assert nisotopepairs == len(self.sig_nu)

        return nisotopepairs

    @cached_property
    def n_trans_nb(self):
        """
        Returns number of transitions n

        """
        ntransitions = len(self.nu[0])

        # all columns should have the same length
        nu_first_dim = self.nu.shape[0]
        assert all([u == nu_first_dim for u in self.nu.shape]), self.nu.shape
        assert all([u == nu_first_dim for u in self.sig_nu.shape]), self.sig_nu.shape
        assert all([u == nu_first_dim for u in self.Xcoeffs.shape]), self.Xcoeffs.shape

        return ntransitions

    @cached_property
    def mu_invm(self):
        """
        Returns difference of the inverse nuclear masses

            mu = 1 / m_a - 1 / m_a'

        where a, a` are isotope pairs.

        """

        assert len(self.m_a) == len(self.m_ap)
        assert all(u != v for u, v in zip(self.m_a, self.m_ap))

        dim = len(self.m_ap)

        return np.divide(np.ones(dim), self.m_a) - np.divide(np.ones(dim), self.m_ap)

    @cached_property
    def mnu(self):
        """
        Generates mass normalised isotope shifts and writes mxn-matrix to file.

            nu / mu

        """

        mnu = np.divide(self.nu.T, self.mu_invm).T
        return mnu

    @cached_property
    def h_np_nucl(self):
        """
        Generates nuclear form factor h for the new physics term.
        h is an m-vector.

        """

        return (self.a_isotope_nb - self.ap_isotope_nb) / self.mu_invm


if __name__ == "__main__":
    Ca = ElemData('Ca')
    print(Ca.nu)
    print(getattr(Ca, 'nu'))
