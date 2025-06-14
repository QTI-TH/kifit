import os
import logging
import numpy as np
from scipy.linalg import lu, det, inv
from math import factorial

from scipy.odr import ODR, Model, RealData
from scipy.stats import linregress

from itertools import permutations, combinations, product
from functools import cache

from kifit.cache_update import update_fct, cached_fct, cached_fct_property
from kifit.user_elements import user_elems

_data_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../kifit_data")
)

###############################################################################

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


def det_64(mat):
    mat_64 = np.array(mat, dtype=np.float64)

    return np.linalg.det(mat_64)


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


# linear King fit
##############################################################################


def linfit(p, x):
    return p[0] * x + p[1]


def get_odr_residuals(p, x, y, sx, sy):

    v = 1 / np.sqrt(1 + p[0] ** 2) * np.array([-p[0], 1])
    z = np.array([x, y]).T
    sz = np.array([np.diag([sx[i] ** 2, sy[i] ** 2]) for i in range(len(x))])

    residuals = np.array([v @ z[i] - p[1] * v[1] for i in range(len(x))])
    sigresiduals = np.sqrt(np.array([v @ sz[i] @ v for i in range(len(x))]))

    return residuals, sigresiduals


def perform_linreg(isotopeshiftdata):  # , reference_transition_index: int = 0):
    """
    Perform linear regression.

    Args:
        isotopeshiftdata (normalised. rows=isotope pairs, columns=trans.)
        reference_transition_index (int, default: first transition)

    Returns:
        betas:       fit parameters (p in linfit)
        sig_betas:   uncertainties on betas
        kperp1s:     Kperp_i1, i=2,...,m (assuming ref. transition index = 0)
        ph1s:        phi_i1, i=2,...,m (assuming ref. transition index = 0)
        sig_kperp1s: uncertainties on kperp1s
        sig_ph1s:    uncertainties on ph1s

    """
    isotopeshiftdata_64 = isotopeshiftdata.astype(np.float64)

    x = isotopeshiftdata_64.T[0]  # [reference_transition_index]
    y = np.delete(isotopeshiftdata_64, 0, axis=1)
    # y = np.delete(isotopeshiftdata, reference_transition_index, axis=1)

    betas = []
    sig_betas = []

    for i in range(y.shape[1]):
        res = linregress(x, y.T[i])
        betas.append(np.array([res.slope, res.intercept]))
        sig_betas.append(np.array([res.stderr, res.intercept_stderr]))

    betas = np.array(betas)
    sig_betas = np.array(sig_betas)

    ph1s = np.arctan(betas.T[0])
    sig_ph1s = np.array(
        [sig_betas[j, 0] / (1 + betas[j, 0]) for j in range(len(betas))]
    )

    kperp1s = betas.T[1] * np.cos(ph1s)
    sig_kperp1s = np.sqrt(
        (sig_betas.T[1] * np.cos(ph1s)) ** 2
        + (betas.T[1] * sig_ph1s * np.sin(ph1s)) ** 2
    )

    return (betas, sig_betas, kperp1s, ph1s, sig_kperp1s, sig_ph1s)


def perform_odr(isotopeshiftdata, sigisotopeshiftdata):
    # , reference_transition_index: int = 0):
    """
    Perform separate orthogonal distance regression for each transition pair.

    Args:
        isotopeshiftdata (normalised. rows=isotope pairs, columns=trans.)
        reference_transition_index (int, default: first transition)

    Returns:
        betas:       fit parameters (p in linfit)
        sig_betas:   uncertainties on betas
        kperp1s:     Kperp_i1, i=2,...,m (assuming ref. transition index = 0)
        ph1s:        phi_i1, i=2,...,m (assuming ref. transition index = 0)
        sig_kperp1s: uncertainties on kperp1s
        sig_ph1s:    uncertainties on ph1s
        cov_kperp1_ph1s: covariance matrices for (kperp1, ph1)
    """
    lin_model = Model(linfit)

    isotopeshiftdata_64 = isotopeshiftdata.astype(np.float64)
    sigisotopeshiftdata_64 = sigisotopeshiftdata.astype(np.float64)

    x = isotopeshiftdata_64.T[0]  # [reference_transition_index]
    y = np.delete(isotopeshiftdata_64, 0, axis=1)
    # y = np.delete(isotopeshiftdata, reference_transition_index, axis=1)

    sigx = sigisotopeshiftdata_64.T[0]
    sigy = np.delete(sigisotopeshiftdata_64, 0, axis=1)
    # sigx = sigisotopeshiftdata.T[reference_transition_index]
    # sigy = np.delete(sigisotopeshiftdata, reference_transition_index, axis=1)

    betas = []
    sig_betas = []
    cov_kperp1_ph1s = []

    for i in range(y.shape[1]):
        data = RealData(x, y.T[i], sx=sigx, sy=sigy.T[i])
        beta_init = np.polyfit(x, y.T[i], 1)
        odr = ODR(data, lin_model, beta0=beta_init)
        out = odr.run()

        # Extract beta and covariance matrix

        beta_out = out.beta
        sig_beta_out = out.sd_beta

        betas.append(beta_out)
        sig_betas.append(sig_beta_out)
        cov_beta = out.cov_beta

        # Calculate ph1 and kperp1
        ph1 = np.arctan(beta_out[0])

        # Derivatives for the delta method
        d_kperp1_d_beta0 = -beta_out[1] * np.sin(ph1)
        d_kperp1_d_beta1 = np.cos(ph1)
        d_ph1_d_beta0 = 1 / (1 + beta_out[0] ** 2)
        d_ph1_d_beta1 = 0

        # Jacobian matrix J
        J = np.array(
            [[d_kperp1_d_beta0, d_kperp1_d_beta1], [d_ph1_d_beta0, d_ph1_d_beta1]]
        )

        # Covariance matrix for (kperp1, ph1)
        cov_kperp1_ph1 = J @ cov_beta @ J.T
        cov_kperp1_ph1s.append(cov_kperp1_ph1)

    betas = np.array(betas)
    sig_betas = np.array(sig_betas)

    ph1s = np.arctan(betas.T[0])
    sig_ph1s = np.arctan(sig_betas.T[0])

    kperp1s = betas.T[1] * np.cos(ph1s)
    sig_kperp1s = np.sqrt(
        (sig_betas.T[1] * np.cos(ph1s)) ** 2
        + (betas.T[1] * sig_ph1s * np.sin(ph1s)) ** 2
    )

    cov_kperp1_ph1s = np.array(cov_kperp1_ph1s)

    return (betas, sig_betas, kperp1s, ph1s, sig_kperp1s, sig_ph1s, cov_kperp1_ph1s)


# Element Collection
##############################################################################


class ElemCollection:

    def __init__(self, elemlist, reference_transitions):
        elem_collection = []
        elem_collection_id = ""

        if reference_transitions is None:
            reference_transitions = np.zeros(len(elemlist), dtype=int)

        for elemstr, ref_trans_ind in zip(elemlist[:-1], reference_transitions[:-1]):
            elem = Elem(str(elemstr), ref_trans_ind)
            elem_collection.append(elem)
            elem_collection_id += elem.id + "_"

        elem = Elem(str(elemlist[-1]), reference_transitions[-1])
        elem_collection.append(elem)
        elem_collection_id += elem.id

        self.elems = elem_collection
        self.id = elem_collection_id

        self.len = len(elem_collection)

        self._init_Xcoeffs()
        # self.check_det_dims(gkpdims, nmgkpdims)

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
        logging.info(
            "All elements have been given X-coefficients with "
            + "compatible mphi values."
        )

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
                    "Determinant methods are only valid for single element."
                )
            else:
                (self.elems[0]).check_det_dims(gkpdims, nmgkpdims, projdims)


# Elem
##############################################################################


class Elem:
    # ADMIN ####################################################################
    # Load raw data from data folder
    # VALID_ELEM = ['Ca']
    VALID_ELEM = user_elems
    INPUT_FILES = [
        "nu",
        "sig_nu",
        "isotopes",
        "binding_energies",
        "Xcoeffs",
        "sig_Xcoeffs",
    ]
    elem_init_atr = [
        "nu_in",
        "sig_nu_in",
        "isotope_data",
        "Eb_data",
        "Xcoeff_data",
        "sig_Xcoeff_data",
    ]

    def __init__(self, element: str, reference_transition_index=0):
        """
        Load all data files associated to element and initialises Elem.
        instance.

        """
        if element not in self.VALID_ELEM:
            raise NameError(
                """Element {} not supported. You may want to add it
                    to src/kifit/user_elems""".format(
                    element
                )
            )

        logging.info("Loading raw data")

        self.id = element
        self.reference_transition_index = reference_transition_index
        self._init_elemdata()
        self._init_masses()
        self._init_Xcoeffs()
        self._init_MC()
        self._init_fit_params()

    def __transform_Xcoeffs(self, val):

        if self.reference_transition_index > 0:

            mphi = val.T[0]
            xs = val[:, 1:]
            x0 = xs.T[self.reference_transition_index]
            xj = np.delete(xs, self.reference_transition_index, axis=1)

            val = np.c_[mphi, x0, xj]

        return val

    def __transform_nus(self, val):

        x = val.T[self.reference_transition_index]
        y = np.delete(val, self.reference_transition_index, axis=1)

        val = np.c_[x, y]

        return val

    def __init_input(self, atr: str, file_path: str):

        logging.info(
            "Loading attribute {} for element {} from {}".format(
                atr, self.id, file_path
            )
        )

        val = np.loadtxt(file_path)

        if (atr == "Xcoeff_data") or (atr == "sig_Xcoeff_data"):

            val = val.reshape(-1, self.ntransitions + 1)
            val = self.__transform_Xcoeffs(val)

        elif (atr == "nu_in") or (atr == "sig_nu_in"):

            val = self.__transform_nus(val)

        setattr(self, atr, val)

    def _init_elemdata(self):
        # load data from elem folder
        file_path = {
            filetype: os.path.join(
                _data_path, self.id, filetype + "_" + self.id + ".dat"
            )
            for filetype in self.INPUT_FILES
        }

        self.__init_input("nu_in", file_path["nu"])
        self.__init_input("sig_nu_in", file_path["sig_nu"])
        self.__init_input("isotope_data", file_path["isotopes"])
        self.__init_input("Eb_data", file_path["binding_energies"])
        self.__init_input("Xcoeff_data", file_path["Xcoeffs"])
        self.__init_input("sig_Xcoeff_data", file_path["sig_Xcoeffs"])

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
            if not np.any(self.Eb_data):
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
                sig_m_a_0**2 + (n_electrons * sig_m_e) ** 2 + sig_Eb @ sig_Eb
            )

            self.m_ap_in = m_ap_0 - n_electrons * m_e + np.sum(Eb)
            self.sig_m_ap_in = np.sqrt(
                sig_m_ap_0**2 + (n_electrons * sig_m_e) ** 2 + sig_Eb @ sig_Eb
            )

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
            raise ValueError(
                """Mediator masses mphi do not match in files with
            X-coefficients and their uncertainties."""
            )
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
        self.dnorm = 1

    def _init_fit_params(self):
        """
        Initialise Kperp1, ph1 to the values obtained through orthogonal
        distance regression on the experimental input data and initialise
        alphaNP to 0.

        """

        (
            _,
            _,
            self.kp1_init,
            self.ph1_init,
            self.sig_kp1_init,
            self.sig_ph1_init,
            self.cov_kperp1_ph1,
        ) = perform_odr(self.nutil_in, self.sig_nutil_in)
        # reference_transition_index=0)

        self.alphaNP_init = 0.0

        self.kp1 = self.kp1_init
        self.ph1 = self.ph1_init
        self.alphaNP = self.alphaNP_init

        self.sig_alphaNP_init = 1

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
        return self.id + "[" + ",".join(list(self.__dict__.keys())) + "]"

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
        if x < 0 or len(self.Xcoeff_data) - 1 < x:
            raise IndexError(f"Index {x} not within permitted range for x.")

        self.x = x
        self.mphi = self.Xcoeff_data[x, 0]
        self.Xvec = self.Xcoeff_data[x, 1:]

        if self.sig_Xcoeff_data[x, 0] != self.mphi:
            raise ValueError(
                """Mediator masses mphi do not match in files with
                    X-coefficients and their uncertainties."""
            )
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

        self.kp1 = thetas[: self.ntransitions - 1]
        self.ph1 = thetas[self.ntransitions - 1 : 2 * self.ntransitions - 2]
        self.alphaNP = thetas[-1]

        if (len(self.kp1) != self.ntransitions - 1) or (
            len(self.ph1) != self.ntransitions - 1
        ):
            raise AttributeError(
                """Passed fit parameters do not have appropriate dimensions"""
            )
        #
        if np.array(
            [(-np.pi / 2 > phij) or (phij > np.pi / 2) for phij in self.ph1]
        ).any():
            raise ValueError(
                """Passed phij values are not within 1st / 4th
            quadrant."""
            )

    @update_fct
    def _update_elem_params(self, vals):
        """
        Set the element parameters

           {m_a, m_ap, nu}

        to the values provided in "vals".

        """
        self.m_a = vals[: self.nisotopepairs]
        self.m_ap = vals[self.nisotopepairs : 2 * self.nisotopepairs]
        self.nu = vals[2 * self.nisotopepairs :].reshape(self.nisotopepairs, -1)

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
        print(
            f"Loaded element {self.id} has {self.nisotopepairs} isotope pairs"
            + f" and {self.ntransitions} transitions."
        )
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
            raise ValueError("""Determinant methods require at least 2 transitions.""")

        for dim in gkpdims:
            if dim < 3:
                raise ValueError(
                    """Generalised King Plot formula is only valid
                for dim >=3."""
                )
            if dim > self.nisotopepairs or dim > self.ntransitions + 1:
                raise ValueError(
                    """GKP dim is larger than dimension of provided
                data."""
                )
            else:
                logging.info(f"GKP dimension {dim} is valid.")

        for dim in nmgkpdims:
            if dim < 3:
                raise ValueError(
                    """No-Mass Generalised King Plot formula is
                only valid for dim >=3."""
                )
            if dim > self.nisotopepairs or dim > self.ntransitions:
                raise ValueError(
                    """NMGKP dim is larger than dimension of provided data."""
                )
            else:
                logging.info(f"Parsed NMGKP dimension {dim} is valid.")

        for dim in projdims:
            if dim < 3:
                raise ValueError("""Projection method is only valid for dim >=3.""")
            if dim > self.nisotopepairs:
                raise ValueError(
                    """proj dim is larger than provided number of isotope pairs."""
                )
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

    @cached_fct_property
    def means_fit_params(self):
        """
        Return all initial values of the fit parameters

           {kp1, ph1, alphaNP}.

        These are computed using an initial linear fit.
        """
        return np.concatenate(
            (self.kp1_init, self.ph1_init, self.alphaNP_init), axis=None
        )

    @cached_fct_property
    def stdevs_input_params(self):
        """
        Return all standard deviations of the input parameters

           {m_a, m_ap, nu}.

        These are provided by the input files.

        """
        return np.concatenate(
            (self.sig_m_a_in, self.sig_m_ap_in, self.sig_nu_in), axis=None
        )

    @cached_fct_property
    def stdevs_fit_params(self):
        """
        Return all standard deviations of the fit parameters

           {kp1, ph1, alphaNP}.

        These are computed using an initial linear fit.
        """
        return np.diag(
            np.concatenate(
                (self.sig_kp1_init, self.sig_ph1_init, self.sig_alphaNP_init), axis=None
            )
        )

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

        return 1 / self.m_a_in - 1 / self.m_ap_in

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

        This is a (nisotopepairs x ntransitions)-matrix.

        """
        return np.divide(self.nu.T, self.muvec).T

    @cached_fct_property
    def sig_nutil_in(self):
        """
        Generate uncertainties on mass normalised isotope shifts.
        Returns a (nisotopepairs x ntransitions)-matrix.

        """

        signutil = np.absolute(
            np.array(
                [
                    [
                        self.nutil_in[a, i]
                        * np.sqrt(
                            (self.sig_nu_in[a, i] / self.nu_in[a, i]) ** 2
                            + 1
                            / (1 / self.m_a_in[a] - 1 / self.m_ap_in[a]) ** 2
                            * (
                                (self.sig_m_a_in[a] / self.m_a_in[a] ** 2) ** 2
                                + (self.sig_m_ap_in[a] / self.m_ap_in[a] ** 2) ** 2
                            )
                        )
                        for i in self.range_i
                    ]
                    for a in self.range_a
                ]
            )
        )

        return signutil

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

        return np.divide(np.ones(dim), self.m_a) - np.divide(np.ones(dim), self.m_ap)

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
        return np.insert(self.kp1, 0, 0.0)

    @cached_fct_property
    def F1(self):
        """
        Field shift vector entering King relation.

           F1 = [1, tan(phi_12), ... , tan(phi_1n)]

        """
        return np.insert(np.tan(self.ph1), 0, 1.0)

    @cached_fct_property
    def F1sq(self):
        """
        Squared norm of field shift vector F1.

        """
        return self.F1 @ self.F1

    @cached_fct_property
    def eF(self):
        return self.F1 / np.sqrt(self.F1sq)

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
        isotopeshiftdata = np.c_[self.nutil_in.T[i], self.nutil_in.T[j]]
        sigisotopeshiftdata = np.c_[self.sig_nutil_in.T[i], self.sig_nutil_in.T[j]]

        betas, _, _, _, _, _, _ = perform_odr(isotopeshiftdata, sigisotopeshiftdata)

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
        return self.Xvec - self.F1 * self.Xvec[0]

    # Construction of the Loglikelihood Function ##############################
    @cached_fct
    def diff_np_term(self, symm: bool):
        """
        Generate the (nisotopepairs x ntransitions)-dimensional new physics term
        starting from theoretical input and fit parameters.

        """
        if symm:
            return self.alphaNP * np.tensordot(
                self.gammatilvec - np.average(self.gammatilvec), self.X1, axes=0
            )

        else:
            np_term_1 = np.tensordot(self.gammatilvec, self.X1, axes=0)
            avg_a_np_term_1 = np.sum(np_term_1 * self.sig_nutil_in, axis=0) / np.sum(
                self.sig_nutil_in, axis=0
            )
            return self.alphaNP * (np_term_1 - avg_a_np_term_1)

    @cached_fct
    def D_a1i(self, a: int, i: int, symm: bool):
        """
        Returns object D_{1j}^a, where a is an isotope pair index and j is a
        transition index.

        """
        if (i == 0) and (a in self.range_a):
            return 0

        elif (i in self.range_j) and (a in self.range_a):
            return (
                self.nutil[a, i]
                - self.F1[i] * self.nutil[a, 0]
                - self.diff_np_term(symm)[a, i]
            )
        else:
            raise IndexError("Index passed to D_a1i is out of range.")

    @cached_fct
    def d_ai(self, a: int, i: int, symm: bool):
        """
        Returns element d_i^{AA'} of the n-vector d^{AA'}.

        """
        if (i == 0) & (a in self.range_a):
            return (
                -1
                / self.F1sq
                * np.sum(
                    np.array(
                        [
                            self.F1[j]
                            * (self.D_a1i(a, j, symm) - self.secph1[j] * self.Kperp1[j])
                            for j in self.range_j
                        ]
                    )
                )
            )

        elif (i in self.range_j) & (a in self.range_a):
            return (
                self.D_a1i(a, i, symm)
                - self.secph1[i] * self.Kperp1[i]
                + self.F1[i] * self.d_ai(a, 0, symm)
            )
        else:
            raise IndexError("Index passed to d_ai is out of range.")

    @cached_fct
    def dmat(self, symm: bool):
        """
        Return full distances matrix.

        """
        return (
            np.array(
                [[self.d_ai(a, i, symm) for i in self.range_i] for a in self.range_a]
            )
            / self.dnorm
        )

    @cached_fct
    def absd(self, symm: bool):
        """
        Returns (nisotopepairs)-vector of Euclidean norms of the
        (ntransitions)-vectors d^{AA'}.

        """
        return np.sqrt(np.diag(self.dmat(symm) @ (self.dmat(symm)).T))

    # Determinant Methods
    ###########################################################################

    # ### GKP #################################################################

    def alphaNP_GKP(self, ainds=[0, 1, 2], iinds=[0, 1]):
        """
        Returns value for alphaNP computed using the Generalised King plot
        formula with the isotope pairs with indices ainds, the transitions with
        indices iinds and the X-coefficients associated to the xth mphi value.

        """
        dim = len(ainds)
        if dim < 3:
            raise ValueError(
                """Generalised King Plot formula is only valid for
            numbers of isotope pairs >=3."""
            )
        if len(iinds) != dim - 1:
            raise ValueError(
                """Generalised King Plot formula requires one
            transition less than the number of isotope pairs."""
            )
        if max(ainds) > self.nisotopepairs:
            raise ValueError(
                """Element %s does not have the required number of
            isotope pairs."""
                % (self.id)
            )
        if max(iinds) > self.ntransitions:
            raise ValueError(
                """Element %s does not have the required number of
            transitions."""
                % (self.id)
            )

        numat = self.nutil[np.ix_(ainds, iinds)]
        mumat = self.mutilvec[np.ix_(ainds)]
        Xmat = self.Xvec[np.ix_(iinds)]  # X-coefficients for a given mphi

        hmat = self.gammatilvec[np.ix_(ainds)]

        vol_data = det_64(np.c_[numat, mumat])

        vol_alphaNP1 = 0
        for i, eps_i in LeviCivita(dim - 1):
            vol_alphaNP1 += eps_i * det_64(
                np.c_[
                    Xmat[i[0]] * hmat,
                    np.array(
                        [numat[:, i[s]] for s in range(1, dim - 1)]
                    ).T,  # numat[:, i[1]],
                    mumat,
                ]
            )

        alphaNP = factorial(dim - 2) * np.array(vol_data / vol_alphaNP1)

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
                """Generalised King Plot formula is only valid for dim >=3."""
            )
        if dim > self.nisotopepairs or dim > self.ntransitions + 1:
            raise ValueError("""dim is larger than dimension of provided data.""")

        voldatlist = []
        vol1st = []
        xindlist = []

        for a_inds, i_inds in product(
            combinations(self.range_a, dim), combinations(self.range_i, dim - 1)
        ):
            # taking into account ordering
            numat = self.nutil[np.ix_(a_inds, i_inds)]
            mumat = self.mutilvec[np.ix_(a_inds)]
            hmat = self.gammatilvec[np.ix_(a_inds)]

            voldatlist.append(det_64(np.c_[numat, mumat]))
            vol1part = []
            xindpart = []
            for i, eps_i in LeviCivita(dim - 1):
                xindpart.append(i_inds[i[0]])  # X always gets first index
                vol1part.append(
                    eps_i
                    * det_64(
                        np.c_[
                            hmat,  # to be multiplied by Xmat[i[0]]
                            np.array([numat[:, i[s]] for s in range(1, dim - 1)]).T,
                            mumat,
                        ]
                    )
                )
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

        alphalist = factorial(dim - 2) * np.array(alphalist)

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
            raise ValueError(
                """Generalised King Plot formula is only valid for
            numbers of isotope pairs >=3."""
            )
        if len(iinds) != dim:
            raise ValueError(
                """Generalised King Plot formula requires same
            number of transitions as isotope pairs."""
            )
        if max(ainds) > self.nisotopepairs:
            raise ValueError(
                """Element %s does not have the required number of
            isotope pairs."""
                % (self.id)
            )
        if max(iinds) > self.ntransitions:
            raise ValueError(
                """Element %s does not have the required number of
            transitions."""
                % (self.id)
            )

        numat = self.nutil[np.ix_(ainds, iinds)]
        Xmat = self.Xvec[np.ix_(iinds)]  # X-coefficients for a given mphi
        hmat = self.gammatilvec[np.ix_(ainds)]

        vol_data = det_64(numat)

        vol_alphaNP1 = 0
        for i, eps_i in LeviCivita(dim):
            vol_alphaNP1 += eps_i * det_64(
                np.c_[
                    Xmat[i[0]] * hmat,
                    np.array([numat[:, i[s]] for s in range(1, dim)]).T,
                ]
            )
        alphaNP = factorial(dim - 1) * np.array(vol_data / vol_alphaNP1)

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
            raise ValueError(
                """No-Mass Generalised King Plot formula is only
            valid for dim >=3."""
            )
        if dim > self.nisotopepairs or dim > self.ntransitions:
            raise ValueError(
                """dim is larger than dimension of provided
            data."""
            )

        voldatlist = []
        vol1st = []
        xindlist = []

        for a_inds, i_inds in product(
            combinations(self.range_a, dim), combinations(self.range_i, dim)
        ):
            # taking into account ordering
            numat = self.nutil[np.ix_(a_inds, i_inds)]
            hmat = self.gammatilvec[np.ix_(a_inds)]

            voldatlist.append(det_64(numat))
            vol1part = []
            xindpart = []
            for i, eps_i in LeviCivita(dim):  # i: indices, eps_i: value
                # continue here: what is GKP doing, is it correct, then NMGKP
                xindpart.append(i_inds[i[0]])  # X always gets first index
                vol1part.append(
                    eps_i
                    * det_64(
                        np.c_[
                            hmat,  # to be multiplied by Xmat[i[0]]
                            np.array([numat[:, i[s]] for s in range(1, dim)]).T,
                        ]
                    )
                )
            vol1st.append(vol1part)
            xindlist.append(xindpart)

        return np.array(voldatlist), np.array(vol1st), xindlist  # , indexlist

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
        alphalist = factorial(dim - 1) * np.array(alphalist)

        return alphalist

    # ### projection method ###################################################

    def pvec(self, v0, v1, v2):

        Dmat = np.c_[v1, v2]

        return (Dmat @ inv(Dmat.T @ Dmat) @ Dmat.T) @ v0

    def Vproj(self, v0, v1, v2):

        pv = self.pvec(v0, v1, v2)

        return np.linalg.norm(v0 - pv) * np.sqrt((v1 @ v1) * (v2 @ v2) - (v1 @ v2) ** 2)

    def alphaNP_proj_Xindep_part(self, ainds=[0, 1, 2], iinds=[0, 1]):
        """
        Returns X-coefficient independent part of alphaNP computed using the
        projection method with the isotope pairs with indices ainds, the
        transitions with indices iinds.

        """
        if len(iinds) != 2:
            raise ValueError(
                """ Projection method is only valid for data sets
            with two transitions."""
            )

        dim = len(ainds)
        if dim < 3:
            raise ValueError(
                """Projection method requires at least 3 isotope
            pairs."""
            )

        mnu1 = np.array([(self.nutil)[a, iinds[0]] for a in ainds])  # (self.nutil.T)[0]
        mnu2 = np.array([(self.nutil)[a, iinds[1]] for a in ainds])  # (self.nutil.T)[1]

        mmu = np.array([self.mutilvec[a] for a in ainds])
        mgamma = np.array([self.gammatilvec[a] for a in ainds])

        Vexp = self.Vproj(mmu, mnu1, mnu2)
        XindepVth = self.Vproj(mmu, mnu1, mgamma)

        return Vexp / XindepVth

    def alphaNP_proj(self, ainds=[0, 1, 2], iinds=[0, 1]):
        """
        Returns value for alphaNP computed using the projection method with the
        isotope pairs with indices ainds, the transitions with indices iinds.

        """
        return self.alphaNP_proj_Xindep_part(ainds=ainds, iinds=iinds) / self.Xji(
            j=iinds[1], i=iinds[0]
        )

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
                """proj dim is larger than provided number of isotope pairs."""
            )

        alphapartlist = []
        xindlist = []

        for a_inds, i_inds in product(
            combinations(self.range_a, dim), combinations(self.range_i, 2)
        ):
            # taking into account ordering
            alphapartlist.append(
                self.alphaNP_proj_Xindep_part(ainds=a_inds, iinds=i_inds)
            )
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
