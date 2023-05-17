import os
import numpy as np
from functools import cache
from functools import cached_property
from loadelemdata.user_elements import user_elems

_data_path = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../data'
))

class ElemData:
    # Load raw data from data folder
    # VALID_ELEM = ['Ca']
    VALID_ELEM = user_elems
    ELEM_FILES = ['nu', 'signu', 'Xvals', 'isotopes']
    CACHE = {}

    def __init__(self, element: str):
        assert element in self.VALID_ELEM, "Element {} not supported".format(element)
        print("Loading raw data")
        # load all data files associated to element
        self.__id = element

        for atr in self.ELEM_FILES:
            file_name = atr + self.__id + '.dat'
            file_path = os.path.join(_data_path, self.__id, file_name)
            print("test: ")
            print(file_path)
            assert os.path.exists(file_path), file_path
            self.__load(atr, file_path)


    def __load(self, atr: str, file_path: str):
        print('loading attribute {} for element {} from {}'.format(atr, self.__id, file_path))
        val = np.loadtxt(file_path)
        setattr(self, atr, val)


    @classmethod
    def load_all(cls):
        """
        loads all elements and returns result as dict
        """
        return {u: cls(u) for u in cls.VALID_ELEM}

    def __repr__(self):
        return self.__id + '[' + ','.join(self.ELEM_FILES) + ']'

    @classmethod
    @cache
    def get(cls, elem: str):
        return cls(elem)


    @cached_property
    def mass_a(self):
        """
        Returns masses of reference isotopes A

        """
        return self.isotopes[2]

    @cached_property
    def mass_ap(self):
        """
        Returns masses of isotopes A'

        """
        return self.isotopes[3]


    @cached_property
    def mu_a_ap(self):
        """
        Returns difference of the inverse nuclear masses

            mu = 1 / m_a - 1 / m_a'

        where a, a` are isotope pairs.

        """

        # we are dealing with an isotope pair ('a' , 'a`')
        m_a = self.mass_a
        m_ap = self.mass_ap

        assert len(m_a) == len(m_ap)
        assert all(u != v for u, v in zip(m_a, m_ap))

        dim = len(m_ap)

        muvec = np.divide(np.ones(dim), m_a) - np.divide(np.ones(dim), m_ap)

        return muvec



    @cached_property
    def reduced_isotope_shifts(self):
        """
        Generates mass normalised isotope shifts and writes mxn-matrix to file.

            nu / mu

        """

        mnu = np.divide(self.nu.T,self.mu_a_ap).T
        return mnu



    @cached_property
    def new_physics_term(self):
        """
        Generates m x (n-1)-dimensional matrix of new physics terms (alphaNP=1) from theoretical input
        """
        isotope_number_a = self.isotopes[0]
        isotope_number_ap = self.isotopes[1]

        nuclear_form_factor = np.divide(
            isotope_number_a - isotope_number_ap,
            self.reduced_isotope_shifts
        )

        X_value = self.Xvals[0]

        return np.multiply(X_value, nuclear_form_factor)

    @cached_property
    def reduced_isotope_shift_covariance_matrix(self):
        """
        Generates (m x m) x (n x n)-dimensional covariance matrix of the reduced
        isotope shifts
        """
        pass


        sigma_mass_a = self.isotopes[4]
        sigma_mass_ap = self.isotopes[5]

        #Snu = [];

        #Snu[AAp,i] = (signu[AAp,i]/mu[AAp])**2;

        #Smu[AAp,BBp,i,j]= (mnu[AAp,i] * sigm[A]

        #self.reduced_isotope_shifts * reduced_isotope_shifts





if __name__ == "__main__":
  Ca = ElemData('Ca')

  print(Ca.nu)
  print(getattr(Ca, 'nu'))
