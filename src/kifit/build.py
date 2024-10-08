import os
import logging
import numpy as np
from itertools import permutations, combinations, product
from functools import cache

from kifit.cache_update import update_fct, cached_fct, cached_fct_property
from kifit.user_elements import user_elems

from kifit.fitools import perform_odr

_data_path = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../kifit_data'
))


# Unit conversions
eV_to_u = 1 / 931494102.42  # conversion of electronvolt to atomic units

# Constants
m_e = 548.579909065e-6  # electron mass in atomic units (u), TIESINGA 2021 (CODATA 2018)
sig_m_e = 1.6e-14  # uncertainty on electron mass in atomic units (u), TIESINGA 2021 (CODATA 2018)


def sec(x: float):
    """
    Compute secant.

    """
    return 1 / np.cos(x)


def Levi_Civita_generator(n):
    """
    Generate indices and signs of non-zero elements of Levi-Civita tensor of
    dimension n.

    """
    if n < 2:
        raise ValueError("Dimension must be at least 2.")

    index_permutations = permutations(range(n))

    for indices in index_permutations:
        sign = 1
        for i in range(n):
            for j in range(i + 1, n):
                if indices[i] > indices[j]:
                    sign *= -1

        yield indices, sign


@cache
def LeviCivita(dim):
    """
    Save Levi-Civita indices and signs for later.

    """
    return list(Levi_Civita_generator(dim))


class ElemCollection:

    def __init__(self, elemlist, gkpdims, nmgkpdims):
        elem_collection = []
        elem_collection_id = ""

        for elemstr in elemlist[:-1]:
            elem = Elem(str(elemstr))
            elem_collection.append(elem)
            elem_collection_id += elem.id + "_"

        elem = Elem(str(elemlist[-1]))
        elem_collection.append(elem)
        elem_collection_id += elem.id

        self.elems = elem_collection
        self.id = elem_collection_id

        self.len = len(elem_collection)

        self._init_Xcoeffs()

        #self.check_det_dims(gkpdims, nmgkpdims)

    def _init_Xcoeffs(self):
        # check whether the Xcoeff are the same
        first_list = np.round(np.asarray(self.elems[0].Xcoeff_data).T[0], decimals=4)

        for elem in self.elems:
            new_list = np.round(np.asarray(elem.Xcoeff_data).T[0], decimals=4)
            if (new_list != first_list).any():
                raise ValueError(
                    "Please prepare data with same mphi values for all "
                    + "elements in the collection."
                )
        logging.info("All elements have been given X-coefficients with "
        + "compatible mphi values.")

        self.mphis = first_list
        self.x_vals = (self.elems[0]).x_vals

    def check_det_dims(self, gkpdims, nmgkpdims, projdims):
        """
        Check whether the demanded generalised King plot and no-mass generalised
        King plot dimensions are compatible with the data associated to the
        first element in the element collection. (N.B.: The determinant methods
        only take one element at a time.

        """
        if not (not gkpdims and not nmgkpdims and not projdims):
            if self.len != 1:
                raise IndexError(
                    "Determinant methods are only valid for single element.")
            else:
                (self.elems[0]).check_det_dims(gkpdims, nmgkpdims, projdims)


class Elem:
    # ADMIN ####################################################################

    # Load raw data from data folder
    # VALID_ELEM = ['Ca']
    VALID_ELEM = user_elems
    INPUT_FILES = [
        'nu', 'sig_nu', 'isotopes', 'binding_energies',
        'Xcoeffs', 'sig_Xcoeffs']
    elem_init_atr = [
        'nu_in', 'sig_nu_in', 'isotope_data', 'Eb_data',
        'Xcoeff_data', 'sig_Xcoeff_data']

    def __init__(self, element: str):
        """
        Load all data files associated to element and initialises Elem.
        instance.

        """
        if element not in self.VALID_ELEM:
            raise NameError("""Element {} not supported. You may want to add it
                    to src/kifit/user_elems""".format(element))

        logging.info("Loading raw data")
        self.id = element
        self._init_elemdata()
        self._init_masses()
        self._init_Xcoeffs()
        self._init_MC()
        self._init_fit_params()

    def __load(self, atr: str, file_type: str, file_path: str):
        logging.info('Loading attribute {} for element {} from {}'.format(
            atr, self.id, file_path))
        val = np.loadtxt(file_path)

        if ((atr == 'Xcoeff_data') or (atr == 'sig_Xcoeff_data')):

            val = val.reshape(-1, self.ntransitions + 1)

        setattr(self, atr, val)

    def _init_elemdata(self):
        # load data from elem folder
        for (i, file_type) in enumerate(self.INPUT_FILES):
            if len(self.INPUT_FILES) != len(self.elem_init_atr):
                raise NameError("""Number of INPUT_FILES does not match number
                of elem_init_atr.""")

            file_name = file_type + '_' + self.id + '.dat'
            file_path = os.path.join(_data_path, self.id, file_name)

            # if not os.path.exists(file_path):
            #     raise ImportError(f"Path {file_path} does not exist.")
            self.__load(self.elem_init_atr[i], file_type, file_path)

    def _init_masses(self):
        """
        Compute nuclear masses from masses of neutral atoms, the electron mass
        and the binding energies, initialise masses.

        """
        if self.id == "Ca_testdata":  # for comparison with Mathematica results
            self.m_a_in = self.isotope_data[1]
            self.sig_m_a_in = self.isotope_data[2]
            self.m_ap_in = self.isotope_data[4]
            self.sig_m_ap_in = self.isotope_data[5]

        else:
            # masses & uncertainties of neutral atoms
            m_a_0 = self.isotope_data[1]
            sig_m_a_0 = self.isotope_data[2]

            m_ap_0 = self.isotope_data[4]
            sig_m_ap_0 = self.isotope_data[5]

            # ionisation energies in eV
            if self.Eb_data.shape[0] == 0:
                Eb = np.array([0])
                sig_Eb = np.array([0])
                n_electrons = 0
            else:
                Eb = self.Eb_data.T[0] * (-1) * eV_to_u
                sig_Eb = self.Eb_data.T[1] * (-1) * eV_to_u
                n_electrons = len(Eb)

            # nuclear masses
            self.m_a_in = m_a_0 - n_electrons * m_e + np.sum(Eb)
            self.sig_m_a_in = np.sqrt(
                sig_m_a_0**2
                + (n_electrons * sig_m_e)**2
                + sig_Eb @ sig_Eb)

            self.m_ap_in = m_ap_0 - n_electrons * m_e + np.sum(Eb)
            self.sig_m_ap_in = np.sqrt(
                sig_m_ap_0**2
                + (n_electrons * sig_m_e)**2
                + sig_Eb @ sig_Eb)

    def _init_Xcoeffs(self):
        """
        Initialise the X coefficients to the set computed for a given mediator
        mass mphi.

        """
        self.mphis = self.Xcoeff_data[:, 0]
        self.x_vals = range(len(self.mphis))

        self.x = 0
        self.mphi = self.Xcoeff_data[self.x, 0]
        self.Xvec = self.Xcoeff_data[self.x, 1:]

        if self.sig_Xcoeff_data[self.x, 0] != self.mphi:
            raise ValueError("""Mediator masses mphi do not match in files with
            X-coefficients and their uncertainties.""")
        else:
            self.sig_Xvec = self.sig_Xcoeff_data[self.x, 1:]

    def _init_MC(self):
        """
        Initialise attributes used in Monte Carlo.

        """
        self.nu = self.nu_in
        self.sig_nu = self.sig_nu_in
        self.m_a = self.m_a_in
        self.m_ap = self.m_ap_in

        # initialise dvec rescaling factor
        self.dnorm = np.mean(self.nu_in / self.sig_nu_in)  # 1.

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
            self.kp1_init, self.ph1_init, self.sig_kp1_init, self.sig_ph1_init, self.cov_kperp1_ph1
        ) = perform_odr(
            self.nutil_in, self.sig_nutil_in,
            reference_transition_index=0)

        self.alphaNP_init = 0.

        self.kp1 = self.kp1_init
        self.ph1 = self.ph1_init
        self.alphaNP = self.alphaNP_init

        self.sig_alphaNP_init = 1

        # self.sig_alphaNP_init = np.absolute(np.max(self.absd) / np.min(
        #     np.tensordot(self.gammatilvec, self.X1[1:], axes=0)))
        # print("sig_alphaNP_init", self.sig_alphaNP_init)

    @update_fct
    def set_alphaNP_init(self, alpha, sigalpha):
        """
        Set alphaNP_init to alpha and sig_alphaNP_init to sigalpha. Useful if

            alphaNP ~ N(0, sig_alphaNP_init)

        is a particularly bad prior.

        """
        if hasattr(alpha, "__len__"):
            raise ValueError("""alphaNP is a scalar.""")
        if hasattr(sigalpha, "__len__"):
            raise ValueError("""sig_alphaNP is a scalar.""")

        logging.info(f"Setting alphaNP_init to     {alpha}.")
        logging.info(f"setting sig_alphaNP_init to {sigalpha}.")

        self.alphaNP_init = alpha
        self.sig_alphaNP_init = sigalpha

    def __repr__(self):
        return self.id + '[' + ','.join(list(self.__dict__.keys())) + ']'

    @classmethod
    def load_all(cls):
        """
        Load all elements and returns result as dict.
        """
        return {u: cls(u) for u in cls.VALID_ELEM}

    @update_fct
    def _update_Xcoeffs(self, x: int):
        """
        Set the X coefficients and their uncertainties to the set computed for.
        a given mediator mass.

        """
        if (x < 0 or len(self.Xcoeff_data) - 1 < x):
            raise IndexError(f"Index {x} not within permitted range for x.")

        self.x = x
        self.mphi = self.Xcoeff_data[x, 0]
        self.Xvec = self.Xcoeff_data[x, 1:]

        if self.sig_Xcoeff_data[x, 0] != self.mphi:
            raise ValueError("""Mediator masses mphi do not match in files with
                    X-coefficients and their uncertainties.""")
        else:
            self.sig_Xvec = self.sig_Xcoeff_data[x, 1:]

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
    def get_dimensions(self):
        """
        Return dimensions of element data in the form

            (nisotopepairs, ntransitions)
        """
        return (self.nisotopepairs, self.ntransitions), self.nu_in

    @cached_fct_property
    def print_dimensions(self):
        """
        Pretty print element data dimensions.

        """
        print()
        print(f"Loaded element {self.id} has {self.nisotopepairs} isotope pairs"
            + f" and {self.ntransitions} transitions.")
        print()

        return 0

    @cached_fct_property
    def print_relative_uncertainties(self):
        """
        Pretty print relative uncertainties on transition and mass measurements.

        """
        print()
        print(f"Relative experimental uncertainties of {self.id} data:")
        print("sig[nu] / nu:     ")
        print(self.sig_nu_in / self.nu_in)
        print()
        print("sig[m_a] / m_a:   ", self.sig_m_a_in / self.m_a_in)
        print("sig[m_ap] / m_ap: ", self.sig_m_ap_in / self.m_ap_in)
        print()

        return 0

    def check_det_dims(self, gkpdims=[], nmgkpdims=[], projdims=[]):
        """
        Check whether Generalised King Plots (No-Mass Generalised King Plots) of
        dimensions gkpdims (nmgkpdims) can be constructed from data of element.

        """
        if self.ntransitions < 2:
            raise ValueError(
                """Determinant methods require at least 2 transitions.""")

        for dim in gkpdims:
            if dim < 3:
                raise ValueError("""Generalised King Plot formula is only valid
                for dim >=3.""")
            if dim > self.nisotopepairs or dim > self.ntransitions + 1:
                raise ValueError("""GKP dim is larger than dimension of provided
                data.""")
            else:
                logging.info(f"GKP dimension {dim} is valid.")

        for dim in nmgkpdims:
            if dim < 3:
                raise ValueError("""No-Mass Generalised King Plot formula is
                only valid for dim >=3.""")
            if dim > self.nisotopepairs or dim > self.ntransitions:
                raise ValueError(
                    """NMGKP dim is larger than dimension of provided data.""")
            else:
                logging.info(f"Parsed NMGKP dimension {dim} is valid.")

        for dim in projdims:
            if dim < 3:
                raise ValueError("""Projection method is only valid for dim >=3.""")
            if dim > self.nisotopepairs:
                raise ValueError(
                    """proj dim is larger than provided number of isotope pairs.""")
            else:
                logging.info(f"Parsed proj dimension {dim} is valid.")

        return (self.nisotopepairs, self.ntransitions)

    @cached_fct_property
    def means_input_params(self):
        """
        Return all mean values of the input parameters

           {m_a, m_ap, nu}.

        These are provided by the input files.

        """
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
    def muvec_in(self):
        """
        Return difference of the inverse nuclear masses

            mu = 1 / m_a - 1 / m_a'

        where a, a' are isotope pairs and m_a, m_a' are the masses given by the
        input files. muvec is an (nisotopepairs)-vector.

        """
        dim = len(self.m_ap_in)

        return np.divide(np.ones(dim), self.m_a_in) - np.divide(np.ones(dim),
                self.m_ap_in)

    @cached_fct_property
    def nutil_in(self):
        """
        Generate mass normalised isotope shifts from input data

            nu / mu.

        This is a (nisotopepairs x ntransitions)-matrix.

        """
        # return np.divide(self.nu_in.T, self.muvec_in).T
        return np.divide(self.nu_in.T, self.muvec_in).T

    @cached_fct_property
    def nutil(self):
        """
        Generate mass normalised isotope shifts

            nu / mu.

        and write (nisotopepairs x ntransitions)-matrix to file.

        """
        return np.divide(self.nu.T, self.muvec).T

    @cached_fct_property
    def sig_nutil_in(self):
        """
        Generate uncertainties on mass normalised isotope shifts and write
        (nisotopepairs x ntransitions)-matrix to file.

        """
        return np.absolute(np.array([[self.nutil_in[a, i]
            * np.sqrt((self.sig_nu_in[a, i] / self.nu_in[a, i])**2
                + (self.m_a_in[a]**2 * self.sig_m_ap_in[a]**2 / self.m_ap_in[a]**2
                    + self.m_ap_in[a]**2 * self.sig_m_a_in[a]**2 / self.m_a_in[a]**2)
                / (self.m_a_in[a] - self.m_ap_in[a])**2) for i in self.range_i] for a
            in self.range_a]))


    # QUANTITIES DERIVED FROM SAMPLED ELEMENT PROPERTIES ######################

    # Nuclear Factors #########################################################

    @cached_fct_property
    def muvec(self):
        """
        Return difference of the inverse nuclear masses

            mu = 1 / m_a - 1 / m_a'

        where a, a` are isotope pairs and m_a, m_a' are the sample masses.
        muvec is an (nisotopepairs)-vector.

        """
        dim = len(self.m_ap)

        return np.divide(np.ones(dim), self.m_a) - np.divide(np.ones(dim),
                self.m_ap)

    @cached_fct_property
    def mutilvec(self):
        """
        Return mu vector, normalised by itself.
        mutilvec is an (nisotopepairs)-vector.

        """
        return np.ones((self.muvec).shape)

    @cached_fct_property
    def gammavec(self):
        """
        Generate nuclear form factor

            gamma^a = gamma^{A_a A_a'} = A_a - A_a',

        for the new physics term. Here A_a and A_a' are the isotope and the
        reference isotope of the a-th isotope pair.

        gammavec is an nisotopepairs-vector.

        """
        return self.a_nisotope - self.ap_nisotope

    @cached_fct_property
    def gammatilvec(self):
        """
        Generate mass-normalised nuclear form factor h for the new physics term.
        h is an nisotopepairs-vector.

        """
        return self.gammavec / self.muvec

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

    # @cached_fct
    def Fji(self, j: int, i: int):
        """
        Return Fji = Fj / Fi for general i, j.

        """
        isotopeshiftdata = np.c_[
            self.nutil_in.T[i],
            self.nutil_in.T[j]
        ]
        sigisotopeshiftdata = np.c_[
            self.sig_nutil_in.T[i],
            self.sig_nutil_in.T[j]
        ]

        betas, _, _, _, _, _, _ = perform_odr(
            isotopeshiftdata, sigisotopeshiftdata, reference_transition_index=0)

        return betas[0, 0]

    # @cached_fct
    def Xji(self, j: int, i: int):
        """
        Return Xji = Xj - Fji Xi for general i, j.

        """
        return self.Xvec[j] - self.Fji(j, i) * self.Xvec[i]

    @cached_fct_property
    def X1(self):
        """
        Return electronic coefficient X_ij of the new physics term.

        """
        return (self.Xvec - self.F1 * self.Xvec[0])

    # Construction of the Loglikelihood Function ##############################
    @cached_fct_property
    def avg_np_term(self):
        """
        Generate the (nisotopepairs x ntransitions)-dimensional new physics term
        starting from theoretical input and fit parameters.

        """
        return self.alphaNP * np.tensordot(
            self.gammatilvec - np.average(self.gammatilvec), self.X1, axes=0)

    @cached_fct
    def D_a1i(self, a: int, i: int):
        """
        Returns object D_{1j}^a, where a is an isotope pair index and j is a
        transition index.

        """
        if ((i == 0) and (a in self.range_a)):
            return 0

        elif ((i in self.range_j) and (a in self.range_a)):
            return (self.nutil[a, i] - self.F1[i] * self.nutil[a, 0]
                    - self.avg_np_term[a, i])
        else:
            raise IndexError('Index passed to D_a1i is out of range.')

    @cached_fct
    def d_ai(self, a: int, i: int):
        """
        Returns element d_i^{AA'} of the n-vector d^{AA'}.

        """
        if ((i == 0) & (a in self.range_a)):
            return (- 1 / self.F1sq * np.sum(np.array([self.F1[j]
                * (self.D_a1i(a, j)
                    - self.secph1[j] * self.Kperp1[j])
                for j in self.range_j])))

        elif ((i in self.range_j) & (a in self.range_a)):
            return (self.D_a1i(a, i)
                    - self.secph1[i] * self.Kperp1[i]
                    + self.F1[i] * self.d_ai(a, 0))
        else:
            raise IndexError('Index passed to d_ai is out of range.')

    @cached_fct_property
    def dmat(self):
        """
        Return full distances matrix.

        """
        return (np.array([[
            self.d_ai(a, i) for i in self.range_i] for a in self.range_a])
            / self.dnorm)

    @cached_fct_property
    def absd(self):
        """
        Returns (nisotopepairs)-vector of Euclidean norms of the
        (ntransitions)-vectors d^{AA'}.

        """
        return np.sqrt(np.diag(self.dmat @ self.dmat.T))

    # Determinant Methods
    ###########################################################################

    # ### GKP #################################################################

    @cached_fct
    def alphaNP_GKP(self, ainds=[0, 1, 2], iinds=[0, 1]):
        """
        Returns value for alphaNP computed using the Generalised King plot
        formula with the isotope pairs with indices ainds, the transitions with
        indices iinds and the X-coefficients associated to the xth mphi value.

        """
        dim = len(ainds)
        if dim < 3:
            raise ValueError("""Generalised King Plot formula is only valid for
            numbers of isotope pairs >=3.""")
        if len(iinds) != dim - 1:
            raise ValueError("""Generalised King Plot formula requires one
            transition less than the number of isotope pairs.""")
        if max(ainds) > self.nisotopepairs:
            raise ValueError("""Element %s does not have the required number of
            isotope pairs.""" % (self.id))
        if max(iinds) > self.ntransitions:
            raise ValueError("""Element %s does not have the required number of
            transitions.""" % (self.id))

        numat = self.nutil[np.ix_(ainds, iinds)]
        mumat = self.mutilvec[np.ix_(ainds)]
        Xmat = self.Xvec[np.ix_(iinds)]  # X-coefficients for a given mphi

        hmat = self.gammatilvec[np.ix_(ainds)]

        vol_data = np.linalg.det(np.c_[numat, mumat])

        vol_alphaNP1 = 0
        for i, eps_i in LeviCivita(dim - 1):
            vol_alphaNP1 += (eps_i * np.linalg.det(np.c_[
                Xmat[i[0]] * hmat,
                np.array([numat[:, i[s]] for s in range(1, dim - 1)]).T,  # numat[:, i[1]],
                mumat]))
        alphaNP = np.math.factorial(dim - 2) * np.array(vol_data / vol_alphaNP1)

        return alphaNP

    def alphaNP_GKP_part(self, dim):
        """
        Prepares the ingredients needed for the computation of alphaNP using the
        Generalised King Plot formula with

           (nisotopepairs, ntransitions) = (dim, dim-1),   dim >= 3.

        The procedure is repeated for all possible combinations of the data that
        fit into this form.

        Since this part of the computation of alphaNP is independent of the
        X-coefficients, it only needs to be evaluated once per element and per
        dim.

        Returns:
            - voldatlist, a numpy array containing the values of the numerator
              for each permutation of the data (dimension: number of combinations),
            - vol1st, numpy array containing the terms in the denominator
              (dimensions: number of combinations, number of epsilon-terms per
              permutation) and
            - xindlist, a list that keeps track of the indices of the required
              X-coefficients (dimensions same as vol1st).

        """
        if dim < 3:
            raise ValueError(
                """Generalised King Plot formula is only valid for dim >=3.""")
        if dim > self.nisotopepairs or dim > self.ntransitions + 1:
            raise ValueError("""dim is larger than dimension of provided data.""")

        voldatlist = []
        vol1st = []
        xindlist = []

        for a_inds, i_inds in product(combinations(self.range_a, dim),
                combinations(self.range_i, dim - 1)):
            # taking into account ordering
            numat = self.nutil[np.ix_(a_inds, i_inds)]
            mumat = self.mutilvec[np.ix_(a_inds)]
            hmat = self.gammatilvec[np.ix_(a_inds)]

            voldatlist.append(np.linalg.det(np.c_[numat, mumat]))
            vol1part = []
            xindpart = []
            for i, eps_i in LeviCivita(dim - 1):
                xindpart.append(i_inds[i[0]])  # X always gets first index
                vol1part.append(
                    eps_i * np.linalg.det(np.c_[
                        hmat,  # to be multiplied by Xmat[i[0]]
                        np.array([numat[:, i[s]] for s in range(1, dim - 1)]).T,
                        mumat]))
            vol1st.append(vol1part)
            xindlist.append(xindpart)

        return np.array(voldatlist), np.array(vol1st), xindlist

    @cached_fct
    def alphaNP_GKP_combinations(self, dim):
        """
        Evaluates alphaNP using the ingredients computed by alphaNP_GKP_part for
        dim=dim.

        If the X-coefficients are varied, this part of the computation of
        alphaNP should be repeated for each set of X-coefficients.

        Returns a list of p alphaNP-values, where p is the number of
        combinations of the data of dimension

           (nisotopepairs, ntransitions) = (dim, dim-1),   dim >= 3.

        """
        voldat, vol1part, xindlist = self.alphaNP_GKP_part(dim)

        alphalist = []

        """ p: alphaNP permutation index and xpinds: X-indices for sample p"""
        for p, xpinds in enumerate(xindlist):
            vol1p = np.array([self.Xvec[xp] for xp in xpinds]) @ (vol1part[p])
            alphalist.append(voldat[p] / vol1p)
        alphalist = np.math.factorial(dim - 2) * alphalist

        return alphalist

    # ### NMGKP ###############################################################
    @cached_fct
    def alphaNP_NMGKP(self, ainds=[0, 1, 2], iinds=[0, 1, 2]):
        """
        Returns value for alphaNP computed using the no-mass Generalised King
        plot formula with the isotope pairs with indices ainds, the transitions
        with indices iinds and the X-coefficients associated to the initialised
        mphi value.

        """
        dim = len(ainds)
        if dim < 3:
            raise ValueError("""Generalised King Plot formula is only valid for
            numbers of isotope pairs >=3.""")
        if len(iinds) != dim:
            raise ValueError("""Generalised King Plot formula requires same
            number of transitions as isotope pairs.""")
        if max(ainds) > self.nisotopepairs:
            raise ValueError("""Element %s does not have the required number of
            isotope pairs.""" % (self.id))
        if max(iinds) > self.ntransitions:
            raise ValueError("""Element %s does not have the required number of
            transitions.""" % (self.id))

        numat = self.nutil[np.ix_(ainds, iinds)]
        Xmat = self.Xvec[np.ix_(iinds)]  # X-coefficients for a given mphi
        hmat = self.gammatilvec[np.ix_(ainds)]

        vol_data = np.linalg.det(numat)

        vol_alphaNP1 = 0
        for i, eps_i in LeviCivita(dim):
            vol_alphaNP1 += (eps_i * np.linalg.det(np.c_[
                Xmat[i[0]] * hmat,
                np.array([numat[:, i[s]] for s in range(1, dim)]).T]))
        alphaNP = np.math.factorial(dim - 1) * np.array(vol_data / vol_alphaNP1)

        return alphaNP

    @cached_fct
    def alphaNP_NMGKP_part(self, dim):
        """
        Prepares the ingredients needed for the computation of alphaNP using the
        No-Mass Generalised King Plot formula with

           (nisotopepairs, ntransitions) = (dim, dim),   dim >= 3.

        The procedure is repeated for all possible combinations of the data that
        fit into this form.

        Since this part of the computation of alphaNP is independent of the
        X-coefficients, it only needs to be evaluated once per element and per
        dim.

        Returns:
            - voldatlist, a numpy array containing the values of the numerator
              for each permutation of the data (dimension: number of combinations),
            - vol1st, numpy array containing the terms in the denominator
              (dimensions: number of combinations, number of epsilon-terms per
              permutation) and
            - xindlist, a list that keeps track of the indices of the required
              X-coefficients (dimensions same as vol1st).

        """
        if dim < 3:
            raise ValueError("""No-Mass Generalised King Plot formula is only
            valid for dim >=3.""")
        if dim > self.nisotopepairs or dim > self.ntransitions:
            raise ValueError("""dim is larger than dimension of provided
            data.""")

        voldatlist = []
        vol1st = []
        xindlist = []

        for a_inds, i_inds in product(combinations(self.range_a, dim),
                combinations(self.range_i, dim)):
            # taking into account ordering
            numat = self.nutil[np.ix_(a_inds, i_inds)]
            hmat = self.gammatilvec[np.ix_(a_inds)]

            voldatlist.append(np.linalg.det(numat))
            vol1part = []
            xindpart = []
            for i, eps_i in LeviCivita(dim):  # i: indices, eps_i: value
                # continue here: what is GKP doing, is it correct, then NMGKP
                xindpart.append(i_inds[i[0]])  # X always gets first index
                vol1part.append(
                    eps_i * np.linalg.det(np.c_[
                        hmat,  # to be multiplied by Xmat[i[0]]
                        np.array([numat[:, i[s]] for s in range(1, dim)]).T]))
            vol1st.append(vol1part)
            xindlist.append(xindpart)

        return np.array(voldatlist), np.array(vol1st), xindlist  #, indexlist

    @cached_fct
    def alphaNP_NMGKP_combinations(self, dim):
        """
        Evaluates alphaNP using the ingredients computed by alphaNP_NMGKP_part
        for dim=dim.

        If the X-coefficients are varied, this part of the computation of
        alphaNP should be repeated for each set of X-coefficients.

        Returns a list of p alphaNP-values, where p is the number of
        combinations of the data of dimension

           (nisotopepairs, ntransitions) = (dim, dim),   dim >= 3.

        """
        voldat, vol1part, xindlist = self.alphaNP_NMGKP_part(dim)

        alphalist = []

        """ p: alphaNP permutation index and xpinds: X-indices for sample p"""
        for p, xpinds in enumerate(xindlist):
            vol1p = np.array([self.Xvec[xp] for xp in xpinds]) @ (vol1part[p])
            alphalist.append(voldat[p] / vol1p)
        alphalist = np.math.factorial(dim - 2) * alphalist

        return alphalist

    # ### projection method ###################################################

    def pvec(self, v1, v2):

        Dmat = np.c_[v1, v2]

        return (Dmat @ np.linalg.inv(Dmat.T @ Dmat) @ Dmat.T) @ self.mutilvec

    def Vproj(self, v0, v1, v2):

        pv = self.pvec(v1, v2)

        return np.linalg.norm(v0 - pv) * np.sqrt(
            (v1 @ v1) * (v2 @ v2) - (v1 @ v2)**2)

    def alphaNP_proj_Xindep_part(self, ainds=[0, 1, 2], iinds=[0, 1]):
        """
        Returns X-coefficient independent part of alphaNP computed using the
        projection method with the isotope pairs with indices ainds, the
        transitions with indices iinds.

        """
        if len(iinds) != 2:
            raise ValueError(""" Projection method is only valid for data sets
            with two transitions.""")

        dim = len(ainds)
        if dim < 3:
            raise ValueError("""Projection method requires at least 3 isotope
            pairs.""")

        mnu1 = (self.nutil.T)[0]
        mnu2 = (self.nutil.T)[1]

        Vexp = self.Vproj(self.mutilvec, mnu1, mnu2)
        XindepVth = self.Vproj(self.mutilvec, mnu1, self.gammatilvec)

        return Vexp / XindepVth

    def alphaNP_proj(self, ainds=[0, 1, 2], iinds=[0, 1]):
        """
        Returns value for alphaNP computed using the projection method with the
        isotope pairs with indices ainds, the transitions with indices iinds.

        """
        return (
            self.alphaNP_proj_Xindep_part(ainds=ainds, iinds=iinds)
            / self.Xji(j=iinds[1], i=iinds[0]))

    @cached_fct
    def alphaNP_proj_part(self, dim):
        """
        Prepares the ingredients needed for the computation of alphaNP using the
        projection method with

           (nisotopepairs, ntransitions) = (dim, 2),   dim >= 3.

        The procedure is repeated for all possible combinations of the data that
        fit into this form.

        Since this part of the computation of alphaNP is independent of the
        X-coefficients, it only needs to be evaluated once per element and per
        dim.

        Returns:
            - list of X-coefficient independent parts of alphaNP, for each
              permutation of the data (dimension: number of combinations),
            - xindlist, a list that keeps track of the indices of the required
              X-coefficients (dimension: number of combinations).

        """
        if dim < 3:
            raise ValueError("""Projection method is only valid for dim >=3.""")
        if dim > self.nisotopepairs:
            raise ValueError(
                """proj dim is larger than provided number of isotope pairs.""")

        alphapartlist = []
        xindlist = []

        for a_inds, i_inds in product(combinations(self.range_a, dim),
                combinations(self.range_i, 2)):
            # taking into account ordering
            alphapartlist.append(self.alphaNP_proj_Xindep_part(ainds=a_inds,
                iinds=i_inds))
            xindlist.append(i_inds)

        return np.array(alphapartlist), xindlist

    @cached_fct
    def alphaNP_proj_combinations(self, dim):
        """
        Evaluates alphaNP using the ingredients computed by
        alphaNP_proj_Xindep_part for dim=dim.

        This part of the computation of alphaNP should be repeated for each set
        of X-coefficients.

        Returns a list of p alphaNP-values, where p is the number of
        combinations of the data of dimension

           (nisotopepairs, 2) = (dim, 2),   dim >= 3.

        """
        alphalist = []

        alphapartlist, xindlist = self.alphaNP_proj_part(dim)

        for p, xind in enumerate(xindlist):

            alphalist.append(alphapartlist[p] / self.Xji(j=xind[1], i=xind[0]))

        return np.array(alphalist)

    # @cached_fct
    # def alphaNP_NMGKP(self, dim):
    #     """
    #     Returns value for alphaNP computed using the no-mass Generalised King
    #     Plot formula.
    #
    #     """
    #     if dim < 3:
    #         raise ValueError("""Generalised King Plot formula is only valid for
    #         dim >=3.""")
    #     if dim > self.nisotopepairs or dim > self.ntransitions:
    #         raise ValueError("""dim is larger than dimension of provided
    #         data.""")
    #
    #     # indexlist = []
    #     alphalist = []
    #
    #     for a_inds, i_inds in product(combinations(self.range_a, dim),
    #             combinations(self.range_i, dim)):
    #
    #         # indexlist.append([a_inds, i_inds])
    #
    #         numat = self.nutil[np.ix_(a_inds, i_inds)]
    #         Xmat = self.Xvec[np.ix_(i_inds)]
    #         hmat = self.gammatilvec[np.ix_(a_inds)]
    #
    #         vol_data = np.linalg.det(numat)
    #
    #         vol_alphaNP1 = 0
    #         for i, eps_i in LeviCivita(dim):
    #             vol_alphaNP1 += (eps_i * np.linalg.det(np.c_[
    #                 Xmat[i[0]] * hmat,
    #                 np.array([numat[:, i[s]] for s in range(1, dim)]).T]))
    #         alphalist.append(vol_data / vol_alphaNP1)
    #
    #     alphalist = np.math.factorial(dim - 1) * np.array(alphalist)
    #
    #     return alphalist   # , indexlist
    #
    # @cached_fct
    # def alphaNP_GKP(self, dim):
    #     """
    #     Returns numpy array of values for alphaNP computed using the Generalised
    #     King Plot formula starting from a data matrix of dimensions
    #
    #        (nisotopepairs, ntransitions) = (dim, dim-1),   dim >= 3.
    #
    #     """
    #     if dim < 3:
    #         raise ValueError("""Generalised King Plot formula is only valid for
    #         dim >=3.""")
    #     if dim > self.nisotopepairs or dim > self.ntransitions + 1:
    #         raise ValueError("""dim is larger than dimension of provided
    #         data.""")
    #
    #     # indexlist = []
    #     alphalist = []
    #
    #     for a_inds, i_inds in product(combinations(self.range_a, dim),
    #             combinations(self.range_i, dim - 1)):
    #
    #         # indexlist.append([a_inds, i_inds])
    #
    #         numat = self.nutil[np.ix_(a_inds, i_inds)]
    #         mumat = self.mutilvec[np.ix_(a_inds)]
    #         Xmat = self.Xvec[np.ix_(i_inds)]
    #         hmat = self.gammatilvec[np.ix_(a_inds)]
    #
    #         vol_data = np.linalg.det(np.c_[numat, mumat])
    #
    #         vol_alphaNP1 = 0
    #         for i, eps_i in LeviCivita(dim - 1):
    #             vol_alphaNP1 += (eps_i * np.linalg.det(np.c_[
    #                 Xmat[i[0]] * hmat,
    #                 np.array([numat[:, i[s]] for s in range(1, dim - 1)]).T,  # numat[:, i[1]],
    #                 mumat]))
    #         alphalist.append(vol_data / vol_alphaNP1)
    #
    #     alphalist = np.math.factorial(dim - 2) * np.array(alphalist)
    #
    #     return alphalist   # , indexlist
    #
    # @cached_fct
    # def alphaNP_NMGKP(self, dim):
    #     """
    #     Returns value for alphaNP computed using the no-mass Generalised King
    #     Plot formula.
    #
    #     """
    #     if dim < 3:
    #         raise ValueError("""Generalised King Plot formula is only valid for
    #         dim >=3.""")
    #     if dim > self.nisotopepairs or dim > self.ntransitions:
    #         raise ValueError("""dim is larger than dimension of provided
    #         data.""")
    #
    #     # indexlist = []
    #     alphalist = []
    #
    #     for a_inds, i_inds in product(combinations(self.range_a, dim),
    #             combinations(self.range_i, dim)):
    #
    #         # indexlist.append([a_inds, i_inds])
    #
    #         numat = self.nutil[np.ix_(a_inds, i_inds)]
    #         Xmat = self.Xvec[np.ix_(i_inds)]
    #         hmat = self.gammatilvec[np.ix_(a_inds)]
    #
    #         vol_data = np.linalg.det(numat)
    #
    #         vol_alphaNP1 = 0
    #         for i, eps_i in LeviCivita(dim):
    #             vol_alphaNP1 += (eps_i * np.linalg.det(np.c_[
    #                 Xmat[i[0]] * hmat,
    #                 np.array([numat[:, i[s]] for s in range(1, dim)]).T]))
    #         alphalist.append(vol_data / vol_alphaNP1)
    #
    #     alphalist = np.math.factorial(dim - 1) * np.array(alphalist)
    #
    #     return alphalist   # , indexlist
