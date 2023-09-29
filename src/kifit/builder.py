import os
import itertools
import numpy as np
import sympy as sp

from sympy import LeviCivita
from sympy.abc import symbols
from sympy import diff
from sympy import Matrix, matrix_multiply_elementwise

from cache_update import update_fct
from cache_update import cached_fct
from cache_update import cached_fct_property
from user_elements import user_elems
from optimizers import Optimizer

_data_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../kifit_data")
)

#%%
def sec(x: float):
    return 1 / np.cos(x)

def levi_civita_tensor(d):
    """
    Return the Levi-Civita tensor as a d-dimensional array

    """
    arr=np.zeros([d for _ in range(d)])
    for x in itertools.permutations(tuple(range(d))):
        mat = np.zeros((d, d), dtype=np.int32)
        for i, j in enumerate(x):
            mat[i][j] = 1
        arr[x]=int(np.linalg.det(mat))
    return arr

def generate_einsum_string(n):
    """
    Return a string of the form 'ij, abc, i, a, b, cj, dk, ...'
    where the stopping point is determined by n.

    """
     
    fixed_transition_indices = ''.join(chr(105 + i) for i in range(n))
                # 'ijk...' for n
    fixed_isotope_pair_indices = ''.join(chr(97 + i) for i in range(n+1))
                # 'abc...' for n+1
    fixed_string = (fixed_transition_indices + ', '
                    + fixed_isotope_pair_indices + ', '
                    + 'i' + ', ' + 'a' + ', ' + 'b')
        
    dynamic_string = []
    for i in range(n-1):
        row_indices = ''.join(chr(99 + i)) #'c'... for n-1
        column_indices = ''.join(chr(106 + i)) #'j'... for n-1
        combi_string = row_indices + column_indices #'cj'
        dynamic_string.append(combi_string) #'cj, dk,...' for n-1
    
    matrix_indices_string = ', '.join(dynamic_string)
    
    einsum_string = fixed_string + ', ' + matrix_indices_string
    
    return einsum_string


class Element:
    # ADMIN ####################################################################

    # Load raw data from data folder
    # VALID_ELEM = ['Ca']
    VALID_ELEM = user_elems
    INPUT_FILES = ["nu", "signu", "isotopes", "Xcoeffs", "sigXcoeffs"]
    elem_init_atr = ["nu", "sig_nu", "isotope_data", "Xcoeffs", "sig_Xcoeffs"]

    OPTIONAL_INPUT_FILES = ["corrnunu", "corrmm", "corrmmp", "corrmpmp", "corrXX"]
    elem_corr_mats = ["corr_nu_nu", "corr_m_m", "corr_m_mp", "corr_mp_mp", "corr_X_X"]

    def __init__(self, element: str):
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

        print("Loading raw data")
        self.id = element
        self._init_elem()
        self._init_corr_mats()
        self._init_fit_params()
        self._init_Xcoeffs()

    def __load(self, atr: str, file_type: str, file_path: str):
        print(
            "Loading attribute {} for element {} from {}".format(
                atr, self.id, file_path
            )
        )
        val = np.loadtxt(file_path)

        if (atr == "Xcoeffs") or (atr == "sig_Xcoeffs"):

            val = val.reshape(-1, self.n_ntransitions + 1)

        setattr(self, atr, val)

    def __set_id_corr_mats(self, atr: str):

        if atr == "corr_nu_nu":
            setattr(
                self,
                atr,
                np.einsum(
                    "ab,ij->aibj",
                    np.identity(self.m_nisotopepairs),
                    np.identity(self.n_ntransitions),
                ),
            )
        elif atr == "corr_m_m":
            if np.all(self.m_a == self.m_a[0]):
                setattr(
                    self,
                    atr,
                    1
                    / (self.m_nisotopepairs * self.m_nisotopepairs)
                    * np.ones((self.m_nisotopepairs, self.m_nisotopepairs)),
                )
            else:
                setattr(self, atr, np.identity(self.m_nisotopepairs))

        elif atr == "corr_mp_mp":
            if np.all(self.m_ap == self.m_ap[0]):
                setattr(
                    self,
                    atr,
                    1
                    / (self.m_nisotopepairs * self.m_nisotopepairs)
                    * np.ones((self.m_nisotopepairs, self.m_nisotopepairs)),
                )
            else:
                setattr(self, atr, np.identity(self.m_nisotopepairs))

        elif atr == "corr_m_mp":
            setattr(self, atr, np.zeros((self.m_nisotopepairs, self.m_nisotopepairs)))

        elif atr == "corr_X_X":
            setattr(self, atr, np.identity(self.n_ntransitions))

    def _init_elem(self):
        for (i, file_type) in enumerate(self.INPUT_FILES):
            if len(self.INPUT_FILES) != len(self.elem_init_atr):
                raise NameError(
                    """Number of INPUT_FILES does not match number
                of elem_init_atr."""
                )

            file_name = file_type + self.id + ".dat"
            file_path = os.path.join(_data_path, self.id, file_name)

            # if not os.path.exists(file_path):
            #     raise ImportError(f"Path {file_path} does not exist.")
            self.__load(self.elem_init_atr[i], file_type, file_path)

    def _init_corr_mats(self):

        corr_mats_to_be_defined = []

        for i, file_type in enumerate(self.OPTIONAL_INPUT_FILES):
            if len(self.OPTIONAL_INPUT_FILES) != len(self.elem_corr_mats):
                raise NameError(
                    """Number of OPTIONAL_INPUT_FILES does not match
                number of elem_corr_mats."""
                )

            file_name = file_type + self.id + ".dat"
            file_path = os.path.join(_data_path, self.id, file_name)

            if os.path.exists(file_path):
                print(
                    "Loading {} for Element {}".format(self.elem_corr_mats[i], self.id)
                )
                self.__load(self.elem_corr_mats[i], file_type, file_path)
            else:
                self.__set_id_corr_mats(self.elem_corr_mats[i])
                corr_mats_to_be_defined.append(self.elem_corr_mats[i])

        if len(corr_mats_to_be_defined) > 0:
            print(
                "Using default values for {} of Element {}.".format(
                    ", ".join(str(cm) for cm in corr_mats_to_be_defined), self.id
                )
            )

    def _init_Xcoeffs(self):
        """
        Initialise the X coefficients to the set computed for a given mediator
        mass mphi.

        """
        self.mphi = self.Xcoeffs[0, 0]
        self.X = self.Xcoeffs[0, 1:]

        if self.sig_Xcoeffs[0, 0] != self.mphi:
            raise ValueError(
                """Mediator masses mphi do not match in files with
            X-coefficients and their uncertainties."""
            )
        else:
            self.sig_X = self.sig_Xcoeffs[0, 1:]

    def _init_fit_params(self):
        self.Kperp1 = np.zeros(self.n_ntransitions)
        self.ph1 = np.zeros(self.n_ntransitions - 1)
        self.alphaNP = 0

    def __repr__(self):
        return self.id + "[" + ",".join(list(self.__dict__.keys())) + "]"

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
        Set the X coefficients and their uncertainties to the set computed for
        a given mediator mass.

        """
        if x < 0 or len(self.Xcoeffs) - 1 < x:
            raise IndexError(f"Index {x} not within permitted range for x.")

        self.mphi = self.Xcoeffs[x, 0]
        self.X = self.Xcoeffs[x, 1:]

        if self.sig_Xcoeffs[x, 0] != self.mphi:
            raise ValueError(
                """Mediator masses mphi do not match in files with
                    X-coefficients and their uncertainties."""
            )
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
        self.Kperp1 = np.insert(thetas[0 : self.n_ntransitions - 1], 0, 0.0)
        self.ph1 = thetas[self.n_ntransitions - 1 : -1]
        self.alphaNP = thetas[-1]

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
        return np.absolute(
            np.array(
                [
                    [
                        self.mu_norm_isotope_shifts[a, i]
                        * np.sqrt(
                            (self.sig_nu[a, i] / self.nu[a, i]) ** 2
                            + (
                                self.m_a[a] ** 2
                                * self.sig_m_ap[a] ** 2
                                / self.m_ap[a] ** 2
                                + self.m_ap[a] ** 2
                                * self.sig_m_a[a] ** 2
                                / self.m_a[a] ** 2
                            )
                            / (self.m_a[a] - self.m_ap[a]) ** 2
                        )
                        for i in self.range_i
                    ]
                    for a in self.range_a
                ]
            )
        )

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
        return self.X - self.F1 * self.X[0]

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
        return np.insert(np.tan(self.ph1), 0, 1.0)

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
        if (i == 0) and (a in self.range_a):
            return 0

        elif (i in self.range_j) and (a in self.range_a):
            return (
                self.nu[a, i]
                - self.F1[i] * self.nu[a, 0]
                - self.mu_aap[a] * self.np_term[a, i]
            )
        else:
            raise IndexError("Index passed to D_a1i is out of range.")

    @cached_fct
    def d_ai(self, a: int, i: int):
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
                            * (
                                self.D_a1i(a, j) / self.mu_aap[a]
                                - self.secph1[j] * self.Kperp1[j]
                            )
                            for j in self.range_j
                        ]
                    )
                )
            )

        elif (i in self.range_j) & (a in self.range_a):
            return (
                self.D_a1i(a, i) / self.mu_aap[a]
                - self.secph1[i] * self.Kperp1[i]
                + self.F1[i] * self.d_ai(a, 0)
            )
        else:
            raise IndexError("Index passed to d_ai is out of range.")

    @cached_fct_property
    def dmat(self):
        """
        Return full distances matrix.

        """
        return np.array([[self.d_ai(a, i) for i in self.range_i] for a in self.range_a])

    @cached_fct_property
    def absd(self):
        """
        Returns m-vector of Euclidean norms of the n-vectors d^{AA'}.

        """
        return np.sqrt(np.diag(self.dmat @ self.dmat.T))

    @cached_fct
    def fDdDnu_ij(self, i: int, j: int):
        """
        Return part of derivative of nu_i^a wrt. nu_j^b that does not depend on.
        isotope indices a = AA', b = BB'.

        """
        if (i == 0) & (j == 0):
            return (
                1
                / self.F1sq
                * np.sum(np.array([self.F1[j] ** 2 for j in self.range_j]))
            )

        elif (i == 0) & (j in self.range_j):
            return -1 / self.F1sq * self.F1[j]

        elif (i in self.range_j) & (j == 0):
            return -1 / self.F1sq * self.F1[i]

        elif (i in self.range_j) & (j in self.range_j):
            res = -1 / self.F1sq * self.F1[i] * self.F1[j]

            if i == j:
                res += 1

            return res

        else:
            return 0

    @cached_fct_property
    def DdDnu(self):
        """
        Return derivative of d wrt. nu.

        """
        fmat = np.array(
            [
                [self.fDdDnu_ij(i, j) if i <= j else 0 for j in self.range_i]
                for i in self.range_i
            ]
        )
        fmat = fmat + fmat.T - np.diag(np.diag(fmat))

        return np.einsum("ab,ij->aibj", np.linalg.inv(np.diag(self.mu_aap)), fmat)

    @cached_fct
    def fDdDmmp_aib(self, a: int, i: int, b: int, merkki: int):

        if i == 0:
            return (
                -merkki
                / self.F1sq
                / self.mu_aap[a] ** 2  # (self.m_a[b] * self.mu_aap[a])**2
                * np.sum(
                    np.array([self.F1[j] * self.D_a1i(a, j) for j in self.range_j])
                )
            )

        elif i in self.range_j:
            return (
                merkki
                / self.mu_aap[a] ** 2  # (self.m_a[b] * self.mu_aap[a])**2 * (
                * (
                    self.D_a1i(a, i)
                    - self.F1[i]
                    / self.F1sq
                    * np.sum(
                        np.array([self.F1[j] * self.D_a1i(a, j) for j in self.range_j])
                    )
                )
            )

    @cached_fct
    def fDdDm_aib(self, a: int, i: int, b: int):
        """
        Return derivative of nu_i^a wrt. m^B, where a = AA' and B is a.
        reference isotope index.

        """
        if a not in self.range_a:
            raise IndexError(
                f"""Isotope pair index {a} passed to fDdDm_aib is
            out of range."""
            )

        if b not in self.range_a:
            raise IndexError(
                f"""Isotope pair index {a} passed t fDdDm_aib is
            out of range."""
            )

        if self.a_nisotope[b] == self.a_nisotope[a]:
            return self.fDdDmmp_aib(a, i, b, 1) / self.m_a[b] ** 2

        elif self.a_nisotope[b] == self.ap_nisotope[a]:
            return self.fDdDmmp_aib(a, i, b, -1) / self.m_a[b] ** 2

        else:
            return 0

    @cached_fct
    def fDdDmp_aib(self, a: int, i: int, b: int):
        """
        Return derivative of nu_i^a wrt. m^{B'}, where a = AA' and B' is a.
        primed isotope index.

        """
        if a not in self.range_a:
            raise IndexError(f"Isotope pair index {a} is out of range.")

        if b not in self.range_a:
            raise IndexError(f"Isotope pair index {a} is out of range.")

        if self.ap_nisotope[b] == self.ap_nisotope[a]:
            return self.fDdDmmp_aib(a, i, b, -1) / self.m_ap[b] ** 2

        elif self.ap_nisotope[b] == self.a_nisotope[a]:
            return self.fDdDmmp_aib(a, i, b, 1) / self.m_ap[b] ** 2

        else:
            return 0

    @cached_fct_property
    def DdDm(self):
        """
        Return derivative of d wrt. m, where m is the vector of reference
        isotope masses.

        """
        return np.array(
            [
                [[self.fDdDm_aib(a, i, b) for b in self.range_a] for i in self.range_i]
                for a in self.range_a
            ]
        )

    @cached_fct_property
    def DdDmp(self):
        """
        Return derivative of d wrt. mp, where mp is the vector of primed
        isotope masses.

        """
        return np.array(
            [
                [[self.fDdDmp_aib(a, i, b) for b in self.range_a] for i in self.range_i]
                for a in self.range_a
            ]
        )

    @cached_fct
    def DdDX_aij(self, a: int, i: int, j: int):
        """
        Return derivative of nu_i^a wrt. X_j, where a = AA' and j is a
        transition index.

        """
        if (a in self.range_a) & (i == 0) & (j == 0):
            return (
                -self.alphaNP
                / self.F1sq
                * self.h_aap[a]
                * np.sum(np.array([self.F1[j] ** 2 for j in self.range_j]))
            )

        elif (a in self.range_a) & (i == 0) & (j in self.range_j):
            return self.alphaNP / self.F1sq * self.h_aap[a] * self.F1[j]

        elif (a in self.range_a) & (i in self.range_j) & (j == 0):
            return (
                self.alphaNP
                * self.h_aap[a]
                * self.F1[i]
                * (
                    1
                    - 1
                    / self.F1sq
                    * np.sum(np.array([self.F1[k] ** 2 for k in self.range_j]))
                )
            )

        elif (a in self.range_a) & (i in self.range_j) & (j in self.range_j):
            res = -1 / self.F1sq * self.F1[i] * self.F1[j]

            if i == j:
                res += 1

            return -self.alphaNP * self.h_aap[a] * res

        else:
            return 0

    @cached_fct_property
    def DdDX(self):
        """
        Return derivative of d wrt. X.

        """
        return np.array(
            [
                [[self.DdDX_aij(a, i, j) for j in self.range_i] for i in self.range_i]
                for a in self.range_a
            ]
        )

    @cached_fct_property
    def cov_nu_nu(self):
        """
        Return covariance matrix of the isotope shift measurements nu, given
        the uncertainties on the isotope shift measurements and their
        correlations.

        """
        return np.einsum("ai,aibj,bj->aibj", self.sig_nu, self.corr_nu_nu, self.sig_nu)

    @cached_fct_property
    def cov_m_m(self):
        """
        Return covariance matrix of the reference isotope mass measurements.
        m_A, given the uncertainties on the isotope mass measurements and their
        correlations.

        """
        return np.einsum("a,ab,b->ab", self.sig_m_a, self.corr_m_m, self.sig_m_a)

    @cached_fct_property
    def cov_m_mp(self):
        """
        Return covariance matrix of the measurements of the reference isotope
        masses m_A and the primed isotope masses, given the experimental
        uncertainties and their correlations.

        """
        return np.einsum("a,ab,b->ab", self.sig_m_a, self.corr_m_mp, self.sig_m_ap)

    @cached_fct_property
    def cov_mp_mp(self):
        """
        Return covariance matrix of the primed isotope mass measurements
        m_Ap, given the experimental uncertainties and their correlations.

        """
        return np.einsum("a,ab,b->ab", self.sig_m_ap, self.corr_mp_mp, self.sig_m_ap)

    @cached_fct_property
    def cov_X_X(self):
        """
        Return covariance matrix of the X coefficients given the theoretical
        uncertainties and their correlations.

        """
        return np.einsum("i,ij,j->ij", self.sig_X, self.corr_X_X, self.sig_X)

    @cached_fct_property
    def cov_d_d(self):
        """
        Return the covariance matrix cov[d,d].

        """
        normald = self.dmat / self.absd[:, None]

        return (
            np.einsum(
                "ai,aick,ckdl,bjdl,bj->ab",
                normald,
                self.DdDnu,
                self.cov_nu_nu,
                self.DdDnu,
                normald,
            )
            + np.einsum(
                "ai,aic,cd,bjd,bj->ab",
                normald,
                self.DdDm,
                self.cov_m_m,
                self.DdDm,
                normald,
            )
            + np.einsum(
                "ai,aic,cd,bjd,bj->ab",
                normald,
                self.DdDm,
                self.cov_m_mp,
                self.DdDmp,
                normald,
            )
            + np.einsum(
                "ai,aic,cd,bjd,bj->ab",
                normald,
                self.DdDmp,
                self.cov_m_mp.T,
                self.DdDm,
                normald,
            )
            + np.einsum(
                "ai,aic,cd,bjd,bj->ab",
                normald,
                self.DdDmp,
                self.cov_mp_mp,
                self.DdDmp,
                normald,
            )
            + np.einsum(
                "ai,aik,kl,bjl,bj->ab",
                normald,
                self.DdDX,
                self.cov_X_X,
                self.DdDX,
                normald,
            )
        )

    @cached_fct_property
    def LL(self):
        """
        Generate the contribution of the element to the negative log-likelihood LL.

        """
        return 1 / 2 * (self.absd @ np.linalg.inv(self.cov_d_d) @ self.absd)
    
# KING PLOT FUNCTIONS #####################################################
    
    @cached_fct_property
    def volume_data_gkp(self):
        """
        Return volume of NLs in mass-normalised isotope shifts for GKP formula.
        
            Add an m-dimensional identity column to mass-normalised data matrix
            and calculate the determinant of the extended matrix.
            
        """
                
        m = len(self.mu_norm_isotope_shifts)
        if m != len(self.mu_norm_isotope_shifts.T) + 1:
            raise ValueError('Wrong shape of data matrix')
            
        datamatrix_extended = np.hstack((self.mu_norm_isotope_shifts,
                                          np.ones(m)[:, np.newaxis]))
        
        return np.linalg.det(datamatrix_extended)
    
    # @cached_fct_property
    @update_fct
    def volume_theory_gkp(self, x: int):
        """
        Return theory volume of NLs for the GKP formula, for alpha_NP=1
        for a given mediator mass
    
        """
        self._update_Xcoeffs(x)
        # self.X = self.Xcoeffs[x, 1:]
        
        if len(self.X) != len(self.mu_norm_isotope_shifts.T):
            raise ValueError('Wrong dimension of X vector')
        
        if len(self.h_aap) != len(self.mu_norm_isotope_shifts):
            raise ValueError('Wrong dimension of h vector')
        # These checks are already done when building the king, does it make
        # sense to do them again?
            
        n = len(self.X)
        matrix_list = [self.mu_norm_isotope_shifts] * (n-1)
        
        vol = np.einsum(generate_einsum_string(n), levi_civita_tensor(n),
                        levi_civita_tensor(n+1), self.X, np.ones(n+1),
                        self.h_aap, *matrix_list)
        norm = np.math.factorial(n-1)
        
        return vol/norm
    
    # @cached_fct_property
    @update_fct
    def alpha_gkp(self, x: int): 
        """
        Return alpha_NP for the GKP formula for a given mediator mass.
    
        """
                
        return self.volume_data_gkp/self.volume_theory_gkp(x)
    
    @cached_fct_property
    def volume_data_gkp_symbolic(self):
        """
        Return symbolic expression of NLs volume from data for the GKP formula.
        
        """
    
        nIP = self.m_nisotopepairs
        nT = self.n_ntransitions
        
        #Define variables
        nu = symbols(' '.join([f'v{j+1}{i+1}' for j in range(nIP)
                               for i in range(nT)]))
        m = symbols(' '.join([f'm0{i}' for i in range(1, nIP+1)]))
        mp = symbols(' '.join([f'm{i}' for i in range(1, nIP+1)]))
        
        #Organize variables in matrices / vectors
        data = Matrix(nIP, nT, nu)
        red_masses = Matrix([[1/(1/m[i] - 1/mp[i])] for i in range(nIP)])
        reduced_data = Matrix([data.row(a)*red_masses[a] for a in range(nIP)])
        
        gkp_square_data = reduced_data.col_insert(nT, Matrix(np.ones(nIP)))
        
        return gkp_square_data.det()
    
    @cached_fct_property
    def volume_theory_gkp_symbolic(self):
        """
        Return symbolic expression of NLs volume from theory for the GKP formula.
        
        """
        n_isotope_pairs = self.m_nisotopepairs
        n_transitions = self.n_ntransitions
        
        # Define symbolic variables
        nu = symbols(' '.join([f'v{j+1}{i+1}' for j in range(n_isotope_pairs)
                               for i in range(n_transitions)]))
        m = symbols(' '.join([f'm0{i}' for i in range(1, n_isotope_pairs+1)]))
        mp = symbols(' '.join([f'm{i}' for i in range(1, n_isotope_pairs+1)]))
        a = symbols(' '.join([f'A0{i}' for i in range(1, n_isotope_pairs+1)]))
        ap = symbols(' '.join([f'A{i}' for i in range(1, n_isotope_pairs+1)]))
        x = symbols(' '.join([f'X{i}' for i in range(1, n_transitions+1)]))
    
        # Organize symbolic variables in matrices and vectors
        data = Matrix(n_isotope_pairs, n_transitions, nu)
        aap = Matrix([[a[i] - ap[i]] for i in range(n_isotope_pairs)])
        red_masses = Matrix([[1/(1/m[i] - 1/mp[i])]
                             for i in range(n_isotope_pairs)])
        hvector = matrix_multiply_elementwise(aap, red_masses)
        reduced_data = Matrix([data.row(a)*red_masses[a]
                               for a in range(n_isotope_pairs)])
    
        # Define indices for transitions and isotope pairs
        transition_indices = symbols(' '.join([chr(8+i)
                                               for i in range(0, n_transitions)]))
        ip_indices = symbols(' '.join([chr(i) for i in range(0, n_isotope_pairs)]))
    
        # Build symbolic expression of NLs volume
        vol_th_sym = 0
        for transition_indices in itertools.product(range(n_transitions),
                                                    repeat=n_transitions):
            for ip_indices in itertools.product(range(n_isotope_pairs),
                                                repeat=n_isotope_pairs):
                base = (LeviCivita(*transition_indices)*LeviCivita(*ip_indices)
                        *x[transition_indices[0]]*hvector[ip_indices[1]])
                for w in range(1, n_transitions):
                    add = reduced_data.col(transition_indices[w])[ip_indices[w+1]]
                    base *= add
                    vol_th_sym += base
    
    
        return vol_th_sym
    
    @cached_fct_property
    def alpha_gkp_symbolic(self):
        """
        Return symbolic expression of alpha_NP for the GKP formula.
        
        """
        return self.volume_data_gkp_symbolic/self.volume_theory_gkp_symbolic
    
    # @cached_fct_property
    @update_fct
    def sig_alpha_gkp_symbolic(self, x: int):
        """
        Return symbolic expression of alpha_NP for the GKP formula.
        
        """
        self._update_Xcoeffs(x)
        
        n_isotope_pairs = self.m_nisotopepairs
        n_transitions = self.n_ntransitions
        
        #Define symbolic variables
        nu = symbols(' '.join([f'v{j+1}{i+1}' for j in range(n_isotope_pairs)
                               for i in range(n_transitions)]))
        m = symbols(' '.join([f'm0{i}' for i in range(1, n_isotope_pairs+1)]))
        mp = symbols(' '.join([f'm{i}' for i in range(1, n_isotope_pairs+1)]))
        x = symbols(' '.join([f'X{i}' for i in range(1, n_transitions+1)]))
        
        derivatives = 0
        for i in range(len(nu)):
            derivatives += (diff(self.alpha_gkp_symbolic, nu[i])**2
                            *self.sig_nu.flatten()[i]**2)
        for i in range(len(mp)):
            derivatives += (diff(self.alpha_gkp_symbolic, mp[i])**2
                            *self.sig_m_a[i]**2)
            for i in range(len(m)):
                derivatives += (diff(self.alpha_gkp_symbolic, m[i])**2
                                *self.sig_m_ap[i]**2)
        for i in range(len(x)):
            derivatives += (diff(self.alpha_gkp_symbolic, x[i])**2
                            *self.sig_X[i]**2)
        
        return sp.sqrt(derivatives)
    
    # @cached_fct_property
    @update_fct
    def sig_alpha_gkp(self, x: int):
        
        self._update_Xcoeffs(x)
        
        n_isotope_pairs = self.m_nisotopepairs
        n_transitions = self.n_ntransitions
        
        #Define symbolic variables
        nu = symbols(' '.join([f'v{j+1}{i+1}' for j in range(n_isotope_pairs)
                                for i in range(n_transitions)]))
        m = symbols(' '.join([f'm0{i}' for i in range(1, n_isotope_pairs+1)]))
        mp = symbols(' '.join([f'm{i}' for i in range(1, n_isotope_pairs+1)]))
        a = symbols(' '.join([f'A0{i}' for i in range(1, n_isotope_pairs+1)]))
        ap = symbols(' '.join([f'A{i}' for i in range(1, n_isotope_pairs+1)]))
        xcoeff = symbols(' '.join([f'X{i}' for i in range(1, n_transitions+1)]))
        
        replacements = []
        replacements += [(nu[i], self.nu.flatten()[i]) for i in 
                          range(self.m_nisotopepairs*self.n_ntransitions)]
        replacements += [(m[i], self.m_a[i]) for i in
                          range(self.m_nisotopepairs)]
        replacements += [(mp[i], self.m_ap[i]) for i in
                          range(self.m_nisotopepairs)]
        replacements += [(xcoeff[i], self.X[i]) for i in range(self.n_ntransitions)]
        replacements += [(a[i], self.a_nisotope[i]) for i in
                          range(self.m_nisotopepairs)]
        replacements += [(ap[i], self.ap_nisotope[i]) for i in
                          range(self.m_nisotopepairs)]
        
        # sig = self.sig_alpha_gkp_symbolic(x)
        
        return self.sig_alpha_gkp_symbolic(x).subs(replacements)
        
        
                          




class ElementsCollection:
    """Collection of elements."""

    def __init__(self, elements=[]):
        """
        Initialize elements collection.

        Args:
            elements (list): list of elements.
        """

        self.elements = elements
        # alphaNP is in common
        self.alphaNP = -5e-11
        # dummy initialization of an optimizer since we need linear fit in collection
        # TO FIX
        self.opt = Optimizer(target_loss=100, max_iterations=100)

    def add(self, element):
        """Add new element to the collection."""
        self.elements.append(element)

    @property
    def get_parameters(self):
        """
        Get all elements parameters as a flatted list of shape:
        [kperp1_elem1, ph1_elem1, kperp1_elem2, ph1_elem2, ..., alphaNP]

        Returns:
            flatten list of parameters of the shape shown above.
        """
        parameters = []
        for elem in self.elements:
            _, _, kperp1, ph1 = self.opt.get_linear_fit_params(
                data=elem.mu_norm_isotope_shifts, reference_transition_idx=0
            )
            parameters.extend(kperp1)
            parameters.extend(ph1)
        # append alphaNP
        parameters.append(self.alphaNP)

        return parameters

    def init_collection(self, path="../kifit_data/"):
        """
        Upload data from `path`.

        Args:
            path (`pathlib.Path`): data path [default "./kifit_data/"].
        """
        # upload elements by folder names
        for element_folder in os.listdir(path):
            if element_folder in user_elems:
                self.add(Element(element_folder))

    def LL(self, parameters=None):
        """
        Build the loss function associated to a collection of elements.

        Args:
            parameters: flatten list of parameters containing Kperp1 and ph1 for all
                the elements in `elements`. The last list item must be alphaNP.
            elements: list of `loadelems.Elem` involved into the experiment.
        """

        if parameters is None:
            parameters = self.get_parameters()

        # get alpha NP
        alphaNP = parameters[-1]
        # initialize index list to be zero
        parameter_index = 0
        # initial loss function value
        ll = 0

        # cycle over the parameters
        for elem in self.elements:
            # number of transition for elem
            n_transitions = elem.n_ntransitions
            # create flatten list of params corresponding to elem's params
            elem_params = list(
                parameters[parameter_index : parameter_index + 2 * (n_transitions - 1)]
            )
            # with alphaNP in the end
            elem_params.append(alphaNP)
            # inject params into the element
            elem._update_fit_params(elem_params)
            # calculating log likelihood
            ll += elem.LL
            # updating index
            parameter_index += 2 * (n_transitions - 1)

        return ll
    
