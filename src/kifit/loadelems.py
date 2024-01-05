import os
import numpy as np
from kifit.cache_update import update_fct
from kifit.cache_update import cached_fct
from kifit.cache_update import cached_fct_property
from kifit.user_elements import user_elems
from kifit.performfit import perform_odr
from scipy.stats import multivariate_normal

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
    elem_init_atr = ['nu_in', 'sig_nu_in', 'isotope_data', 'Xcoeff_data',
        'sig_Xcoeff_data']

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
        self._init_Xcoeffs()
        self._init_fit_params()

    def __load(self, atr: str, file_type: str, file_path: str):
        print('Loading attribute {} for element {} from {}'.format(
            atr, self.id, file_path))
        val = np.loadtxt(file_path)

        if ((atr == 'Xcoeff_data') or (atr == 'sig_Xcoeff_data')):

            val = val.reshape(-1, self.ntransitions + 1)

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

        setattr(self, 'nu', self.nu_in)
        setattr(self, 'sig_nu', self.sig_nu_in)

        setattr(self, 'm_a_in', self.isotope_data[1])
        setattr(self, 'm_a', self.m_a_in)
        setattr(self, 'sig_m_a_in', self.isotope_data[2])

        setattr(self, 'm_ap_in', self.isotope_data[4])
        setattr(self, 'm_ap', self.m_ap_in)
        setattr(self, 'sig_m_ap_in', self.isotope_data[5])


    def _init_Xcoeffs(self):
        """
        Initialise the X coefficients to the set computed for a given mediator
        mass mphi.

        """
        self.mphi = self.Xcoeff_data[0, 0]
        self.X = self.Xcoeff_data[0, 1:]

        if self.sig_Xcoeff_data[0, 0] != self.mphi:
            raise ValueError("""Mediator masses mphi do not match in files with
            X-coefficients and their uncertainties.""")
        else:
            self.sig_X = self.sig_Xcoeff_data[0, 1:]

    def _init_fit_params(self):
        """
        Initialise Kperp1, ph1 to the values obtained through orthogonal
        distance regression on the experimental input data and initialise
        alphaNP to 0.

        """

        # self.Kperp1 = np.zeros(self.ntransitions)
        # self.ph1 = np.zeros(self.ntransitions - 1)
        (
            _, _,
            self.kp1_init, self.ph1_init, self.sig_kp1_init, self.sig_ph1_init
        ) = perform_odr(
            self.mu_norm_isotope_shifts_in, self.sig_mu_norm_isotope_shifts_in,
            reftrans_index=0)

        self.alphaNP_init = 0.

        self.kp1 = self.kp1_init
        self.ph1 = self.ph1_init
        self.alphaNP = self.alphaNP_init

        self.sig_alphaNP_init = np.absolute(np.max(self.absd) / np.min(
            np.tensordot(self.h_aap, self.X1[1:], axes=0)))

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
        if (x < 0 or len(self.Xcoeff_data) - 1 < x):
            raise IndexError(f"Index {x} not within permitted range for x.")

        self.mphi = self.Xcoeff_data[x, 0]
        self.X = self.Xcoeff_data[x, 1:]

        if self.sig_Xcoeff_data[x, 0] != self.mphi:
            raise ValueError("""Mediator masses mphi do not match in files with
                    X-coefficients and their uncertainties.""")
        else:
            self.sig_X = self.sig_Xcoeff_data[x, 1:]

    @update_fct
    def _update_fit_params(self, thetas):
        """
        Set the fit parameters

           {kp1, ph1, alphaNP}

        where kp1 and ph1 are (n-1)-vectors and alphaNP is a scalar,
        to the values provided in the (2 * (n-1) + 1)-dimensional vector
        "thetas".

        """
        # if ((len(thetas[0]) != self.ntransitions - 1)
        #         or (len(thetas[1]) != self.ntransitions - 1)):
        #     raise AttributeError("""Passed fit parameters do not have
        #     appropriate dimensions""")
        #
        # if np.array([(-np.pi / 2 > phij) or (phij > np.pi / 2) for phij in thetas[1]]).any():
        #     raise ValueError("""Passed phij values are not within 1st / 4th
        #     quadrant.""")

        self.kp1 = thetas[:self.ntransitions - 1]
        self.ph1 = thetas[self.ntransitions - 1: 2 * self.ntransitions - 2]
        self.alphaNP = thetas[-1]

        if ((len(self.kp1) != self.ntransitions - 1) or (len(self.ph1) != self.ntransitions - 1)):
            raise AttributeError("""Passed fit parameters do not have appropriate dimensions""")
        #
        if np.array([(-np.pi / 2 > phij) or (phij > np.pi / 2) for phij in self.ph1]).any():
            raise ValueError("""Passed phij values are not within 1st / 4th
            quadrant.""")


    @update_fct
    def _update_elem_params(self, vals):
        """
        Set the element parameters

           {m_a, m_ap, nu}

        to the values provided in "vals".

        """
        self.m_a = vals[:self.nisotopepairs]
        self.m_ap = vals[self.nisotopepairs:2 * self.nisotopepairs]
        self.nu = vals[2 * self.nisotopepairs:].reshape(self.nisotopepairs, -1)


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
    def nisotopepairs(self):
        """
        Return number of isotope pairs

        """
        return len(self.a_nisotope)

    @cached_fct_property
    def ntransitions(self):
        """
        Return number of transitions

        """
        return self.nu_in.shape[1]

    @cached_fct_property
    def means_input_params(self):
        """
        Return all mean values of the input parameters

           {m_a, m_ap, nu}.

        These are provided by the input files.

        """

        # print(self.nu_in.flatten())
        # print(self.m_a_in)
        return np.concatenate((self.m_a_in, self.m_ap_in, self.nu_in), axis=None)
        # import sys
        # sys.exit()
        # return np.array([self.m_a_in, self.m_ap_in, self.nu_in.flatten()]).reshape(-1)

    @cached_fct_property
    def means_fit_params(self):
        """
        Return all initial values of the fit parameters

           {kp1, ph1, alphaNP}.

        These are computed using an initial linear fit.
        """
        # return np.array([self.kp1_init, self.ph1_init, self.alphaNP_init]).reshape(-1)
        return np.concatenate((self.kp1_init, self.ph1_init, self.alphaNP_init), axis=None)

    @cached_fct_property
    def stdevs_input_params(self):
        """
        Return all standard deviations of the input parameters

           {m_a, m_ap, nu}.

        These are provided by the input files.

        """
        # return np.array([self.sig_m_a_in, self.sig_m_ap_in, self.sig_nu_in.flatten()]).reshape(-1)
        return np.concatenate((self.sig_m_a_in, self.sig_m_ap_in, self.sig_nu_in), axis=None)

    @cached_fct_property
    def stdevs_fit_params(self):
        """
        Return all standard deviations of the fit parameters

           {kp1, ph1, alphaNP}.

        These are computed using an initial linear fit.
        """
        # return np.array([self.sig_kp1_init, self.sig_ph1_init, self.sig_alphaNP_init]).reshape(-1)
        return np.diag(np.concatenate((self.sig_kp1_init, self.sig_ph1_init,
            self.sig_alphaNP_init), axis=None))

    @cached_fct_property
    def range_a(self):
        """
        Returns range of isotope indices
           [0, 2, ...., m-1]
        """
        return np.arange(self.nisotopepairs)

    @cached_fct_property
    def range_i(self):
        """
        Returns range of transition indices
           [0, 2, ...., n-1]
        """
        return np.arange(self.ntransitions)

    @cached_fct_property
    def range_j(self):
        """
        Returns range of indices of transitions that are not reference
        transitions.
           [1, ...., n-1]
        """
        return np.arange(1, self.ntransitions)

    # QUANTITIES DERIVED FROM INPUT ELEMENT PROPERTIES #########################

    @cached_fct_property
    def mu_aap_in(self):
        """
        Return difference of the inverse nuclear masses

            mu = 1 / m_a - 1 / m_a'

        where a, a` are isotope pairs and m_a, m_a' are the masses given by the
        input files. mu_aap is an (nisotopepairs)-vector.

        """
        dim = len(self.m_ap_in)

        return np.divide(np.ones(dim), self.m_a_in) - np.divide(np.ones(dim),
                self.m_ap_in)

    @cached_fct_property
    def mu_norm_isotope_shifts_in(self):
        """
        Generate mass normalised isotope shifts from input data

            nu / mu

        and write (nisotopepairs x ntransitions)-matrix to file.

        """
        return np.divide(self.nu_in.T, self.mu_aap_in).T

    @cached_fct_property
    def sig_mu_norm_isotope_shifts_in(self):
        """
        Generate uncertainties on mass normalised isotope shifts and write
        (nisotopepairs x ntransitions)-matrix to file.

        """
        return np.absolute(np.array([[self.mu_norm_isotope_shifts_in[a, i]
            * np.sqrt((self.sig_nu_in[a, i] / self.nu_in[a, i])**2
                + (self.m_a_in[a]**2 * self.sig_m_ap_in[a]**2 / self.m_ap_in[a]**2
                    + self.m_ap_in[a]**2 * self.sig_m_a_in[a]**2 / self.m_a_in[a]**2)
                / (self.m_a_in[a] - self.m_ap_in[a])**2) for i in self.range_i] for a
            in self.range_a]))


    # QUANTITIES DERIVED FROM SAMPLED ELEMENT PROPERTIES ######################

    # Nuclear Factors #########################################################

    @cached_fct_property
    def mu_aap(self):
        """
        Return difference of the inverse nuclear masses

            mu = 1 / m_a - 1 / m_a'

        where a, a` are isotope pairs and m_a, m_a' are the sample masses.
        mu_aap is an (nisotopepairs)-vector.

        """
        dim = len(self.m_ap)

        return np.divide(np.ones(dim), self.m_a) - np.divide(np.ones(dim),
                self.m_ap)

    @cached_fct_property
    def h_aap(self):
        """
        Generate nuclear form factor h for the new physics term.
        h is an m-vector.

        """
        return (self.a_nisotope - self.ap_nisotope) / self.mu_aap

    # Electronic Factors ######################################################
    @cached_fct_property
    def Kperp1(self):
        """
        Component of mass shift vector that is perpendicular to the King line.

        """
        return np.insert(self.kp1, 0, 0.)

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
    def X1(self):
        """
        Return electronic coefficient X_ij of the new physics term.

        """
        return (self.X - self.F1 * self.X[0])

    # Construction of the Loglikelihood Function ##############################
    @cached_fct_property
    def np_term(self):
        """
        Generate the (nisotopepairs x ntransitions)-dimensional new physics term
        starting from theoretical input and fit parameters.

        """
        return np.absolute(self.alphaNP * np.tensordot(self.h_aap, self.X1,
            axes=0))

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
        return np.array([[self.d_ai(a, i) for i in self.range_i] for a in self.range_a])

    @cached_fct_property
    def absd(self):
        """
        Returns (nisotopepairs)-vector of Euclidean norms of the
        (ntransitions)-vectors d^{AA'}.

        """
        return np.sqrt(np.diag(self.dmat @ self.dmat.T))
