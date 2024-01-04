import os
import numpy as np
from kifit.cache_update import update_fct
from kifit.cache_update import cached_fct
from kifit.cache_update import cached_fct_property
from kifit.user_elements import user_elems
from kifit.optimizers import Optimizer

_data_path = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../kifit_data'
))


def sec(x: float):
    return 1 / np.cos(x)


class Elem:
    # ADMIN ####################################################################

    # Load raw data from data folder
    # VALID_ELEM = ['Ca']
    VALID_ELEM = user_elems
    INPUT_FILES = ['nu', 'signu', 'isotopes', 'Xcoeffs', 'sigXcoeffs']
    elem_init_atr = ['nu', 'sig_nu', 'isotope_data', 'Xcoeffs', 'sig_Xcoeffs']

    def __init__(self, element: str):
        """
        Load all data files associated to element and initialises Elem.
        instance.

        """
        if element not in self.VALID_ELEM:
            raise NameError("""Element {} not supported. You may want to add it
                    to src/kifit/user_elems""".format(element))

        print("Loading raw data")
        self.id = element
        self._init_elem()
        self._init_fit_params()
        self._init_Xcoeffs()

    def __load(self, atr: str, file_type: str, file_path: str):
        print('Loading attribute {} for element {} from {}'.format(
            atr, self.id, file_path))
        val = np.loadtxt(file_path)

        if ((atr == 'Xcoeffs') or (atr == 'sig_Xcoeffs')):

            val = val.reshape(-1, self.n_ntransitions + 1)

        setattr(self, atr, val)

    def _init_elem(self):
        for (i, file_type) in enumerate(self.INPUT_FILES):
            if len(self.INPUT_FILES) != len(self.elem_init_atr):
                raise NameError("""Number of INPUT_FILES does not match number
                of elem_init_atr.""")

            file_name = file_type + self.id + '.dat'
            file_path = os.path.join(_data_path, self.id, file_name)

            # if not os.path.exists(file_path):
            #     raise ImportError(f"Path {file_path} does not exist.")
            self.__load(self.elem_init_atr[i], file_type, file_path)

    def _init_Xcoeffs(self):
        """
        Initialise the X coefficients to the set computed for a given mediator
        mass mphi.

        """
        self.mphi = self.Xcoeffs[0, 0]
        self.X = self.Xcoeffs[0, 1:]

        if self.sig_Xcoeffs[0, 0] != self.mphi:
            raise ValueError("""Mediator masses mphi do not match in files with
            X-coefficients and their uncertainties.""")
        else:
            self.sig_X = self.sig_Xcoeffs[0, 1:]

    def _init_fit_params(self):
        self.Kperp1 = np.zeros(self.n_ntransitions)
        self.ph1 = np.zeros(self.n_ntransitions - 1)
        self.alphaNP = 0

    def __repr__(self):
        return self.id + '[' + ','.join(list(self.__dict__.keys())) + ']'

    @classmethod
    def load_all(cls):
        """
        Load all elements and returns result as dict.
        """
        return {u: cls(u) for u in cls.VALID_ELEM}

    @classmethod
    def get(cls, elem: str):
        return cls(elem)

    @update_fct
    def _update_Xcoeffs(self, x: int):
        """
        Set the X coefficients and their uncertainties to the set computed for.
        a given mediator mass.

        """
        if (x < 0 or len(self.Xcoeffs) - 1 < x):
            raise IndexError(f"Index {x} not within permitted range for x.")

        self.mphi = self.Xcoeffs[x, 0]
        self.X = self.Xcoeffs[x, 1:]

        if self.sig_Xcoeffs[x, 0] != self.mphi:
            raise ValueError("""Mediator masses mphi do not match in files with
                    X-coefficients and their uncertainties.""")
        else:
            self.sig_X = self.sig_Xcoeffs[x, 1:]

    @update_fct
    def _update_fit_params(self, thetas):
        """
        Set the fit parameters

           thetas = {kperp1, ph1, alphaNP},

        where kperp1 and ph1 are (n-1)-vectors and alphaNP is a scalar, to the
        values provided in "thetas".

        """
        if ((len(thetas[0]) != self.n_ntransitions - 1)
                or (len(thetas[1]) != self.n_ntransitions - 1)):
            raise AttributeError("""Passed fit parameters do not have
            appropriate dimensions""")

        if np.array([(-np.pi / 2 > phij) or (phij > np.pi / 2) for phij in thetas[1]]).any():
            raise ValueError("""Passed phij values are not within 1st / 4th
            quadrant.""")

        self.Kperp1 = np.insert(thetas[0], 0, 0.)
        self.ph1 = thetas[1]
        self.alphaNP = thetas[2]

        print("new Kperp1 ", thetas[0])
        print("new ph1    ", thetas[1])
        print("new alphaNP", thetas[2])

    # ELEM PROPERTIES ##########################################################

    @cached_fct_property
    def a_nisotope(self):
        """
        Return isotope numbers A

        """
        return self.isotope_data[0]

    @cached_fct_property
    def ap_nisotope(self):
        """
        Return isotope numbers A'

        """
        return self.isotope_data[3]

    @cached_fct_property
    def m_a(self):
        """
        Return masses of reference isotopes A

        """
        return self.isotope_data[1]

    @cached_fct_property
    def m_ap(self):
        """
        Return masses of isotopes A'

        """
        return self.isotope_data[4]

    @cached_fct_property
    def sig_m_a(self):
        """
        Return uncertainties on masses of reference isotopes A

        """
        return self.isotope_data[2]

    @cached_fct_property
    def sig_m_ap(self):
        """
        Return uncertainties on masses of isotopes A'

        """
        return self.isotope_data[5]

    @cached_fct_property
    def m_nisotopepairs(self):
        """
        Return number of isotope pairs m

        """
        return len(self.a_nisotope)

    @cached_fct_property
    def n_ntransitions(self):
        """
        Return number of transitions n

        """
        return self.nu.shape[1]

    @cached_fct_property
    def mu_aap(self):
        """
        Return difference of the inverse nuclear masses

            mu = 1 / m_a - 1 / m_a'

        where a, a` are isotope pairs.
        mu_aap is an m-vector.

        """
        dim = len(self.m_ap)

        return np.divide(np.ones(dim), self.m_a) - np.divide(np.ones(dim), self.m_ap)

    @cached_fct_property
    def mu_norm_isotope_shifts(self):
        """
        Generate mass normalised isotope shifts and write mxn-matrix to file.

            nu / mu

        """
        return np.divide(self.nu.T, self.mu_aap).T

    @cached_fct_property
    def sig_mu_norm_isotope_shifts(self):
        """
        Generate uncertainties on mass normalised isotope shifts and write
        (m x n)-matrix to file.

        """
        return np.absolute(np.array([[self.mu_norm_isotope_shifts[a, i]
            * np.sqrt((self.sig_nu[a, i] / self.nu[a, i])**2
                + (self.m_a[a]**2 * self.sig_m_ap[a]**2 / self.m_ap[a]**2
                    + self.m_ap[a]**2 * self.sig_m_a[a]**2 / self.m_a[a]**2)
                / (self.m_a[a] - self.m_ap[a])**2) for i in self.range_i] for a
            in self.range_a]))

    # CONSTRUCTING THE LOG-LIKELIHOOD FUNCTION ################################

    @cached_fct_property
    def h_aap(self):
        """
        Generate nuclear form factor h for the new physics term.
        h is an m-vector.

        """
        return (self.a_nisotope - self.ap_nisotope) / self.mu_aap

    @cached_fct_property
    def X1(self):
        """
        Return electronic coefficient X_ij of the new physics term.

        """
        return (self.X - self.F1 * self.X[0])

    @cached_fct_property
    def np_term(self):
        """
        Generate the (m x n)-dimensional new physics term starting from
        theoretical input and fit parameters.

        """
        return self.alphaNP * np.tensordot(self.h_aap, self.X1, axes=0)

    @cached_fct_property
    def F1(self):
        """
        Field shift vector entering King relation.

           F1 = [1, tan(phi_12), ... , tan(phi_1n)]

        """
        return np.insert(np.tan(self.ph1), 0, 1.)

    @cached_fct_property
    def F1sq(self):
        """
        Squared norm of field shift vector F1.

        """
        return self.F1 @ self.F1

    @cached_fct_property
    def secph1(self):
        """
        Return secant of angle phi_ij (sec(phi_ij)).

        """
        return np.insert(sec(self.ph1), 0, 0)

    @cached_fct_property
    def range_a(self):
        """
        Returns range of isotope indices
           [0, 2, ...., m-1]
        """
        return np.arange(self.m_nisotopepairs)

    @cached_fct_property
    def range_i(self):
        """
        Returns range of transition indices
           [0, 2, ...., n-1]
        """
        return np.arange(self.n_ntransitions)

    @cached_fct_property
    def range_j(self):
        """
        Returns range of indices of transitions that are not reference
        transitions.
           [1, ...., n-1]
        """
        return np.arange(1, self.n_ntransitions)

    @cached_fct
    def D_a1i(self, a: int, i: int):
        """
        Returns object D_{1j}^a, where a is an isotope pair index and j is a
        transition index.

        """
        if ((i == 0) and (a in self.range_a)):
            return 0

        elif ((i in self.range_j) and (a in self.range_a)):
            return (self.nu[a, i] - self.F1[i] * self.nu[a, 0]
                    - self.mu_aap[a] * self.np_term[a, i])
        else:
            raise IndexError('Index passed to D_a1i is out of range.')

    @cached_fct
    def d_ai(self, a: int, i: int):
        """
        Returns element d_i^{AA'} of the n-vector d^{AA'}.

        """
        if ((i == 0) & (a in self.range_a)):
            return - 1 / self.F1sq * np.sum(np.array([self.F1[j]
                * (self.D_a1i(a, j) / self.mu_aap[a]
                    - self.secph1[j] * self.Kperp1[j])
                for j in self.range_j]))

        elif ((i in self.range_j) & (a in self.range_a)):
            return (self.D_a1i(a, i) / self.mu_aap[a]
                    - self.secph1[i] * self.Kperp1[i]
                    + self.F1[i] * self.d_ai(a, 0))
        else:
            raise IndexError('Index passed to d_ai is out of range.')

    @cached_fct_property
    def dmat(self):
        """
        Return full distances matrix.

        """
        return np.array([[self.d_ai(a, i) for i in self.range_i] for a in
            self.range_a])

    @cached_fct_property
    def absd(self):
        """
        Returns m-vector of Euclidean norms of the n-vectors d^{AA'}.

        """
        return np.sqrt(np.diag(self.dmat @ self.dmat.T))
