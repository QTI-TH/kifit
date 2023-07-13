import os
import numpy as np
from qiss.cache_update import update_fct
from qiss.cache_update import cached_fct
from qiss.cache_update import cached_fct_property
from qiss.user_elements import user_elems

_data_path = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../qiss_data'
))


def sec(x: float):
    return 1 / np.cos(x)


class Elem:
    # ADMIN ####################################################################

    # Load raw data from data folder
    # VALID_ELEM = ['Ca']
    VALID_ELEM = user_elems
    INPUT_FILES = ['nu', 'signu', 'isotopes', 'Xcoeffs', 'sigXcoeffs']
    elem_init_atr = ['nu', 'sig_nu', 'isotope_data', 'Xcoeffs',
    'sig_Xcoeffs']

    OPTIONAL_INPUT_FILES = ['corrnunu', 'corrmm', 'corrmmp', 'corrmpmp',
    'corrXX']
    elem_corr_mats = ['corr_nu_nu', 'corr_m_m', 'corr_m_mp', 'corr_mp_mp',
    'corr_X_X']

    def __init__(self, element: str):
        """
        Load all data files associated to element and initialises Elem.
        instance.

        """
        if element not in self.VALID_ELEM:
            raise NameError("Element {} not supported".format(element))

        print("Loading raw data")
        self.id = element
        self._init_elem()
        self._init_corr_mats()
        self._init_fit_params()
        self._init_Xcoeffs()

    def __load(self, atr: str, file_type: str, file_path: str):
        print('Loading attribute {} for element {} from {}'.format(
            atr, self.id, file_path))
        val = np.loadtxt(file_path)

        if ((atr == 'Xcoeffs') or (atr == 'sig_Xcoeffs')):
            val = val.reshape(-1, self.n_ntransitions)

        setattr(self, atr, val)

    def __set_id_corr_mats(self, atr: str):

        if (atr == 'corr_nu_nu'):
            setattr(self, atr,
                    np.einsum('ab,ij->aibj',
                        np.identity(self.m_nisotopepairs),
                        np.identity(self.n_ntransitions)))
        elif (atr == 'corr_m_m'):
            if np.all(self.m_a == self.m_a[0]):
                setattr(self, atr,
                        1 / (self.m_nisotopepairs * self.m_nisotopepairs)
                        * np.ones((self.m_nisotopepairs,
                    self.m_nisotopepairs)))
            else:
                setattr(self, atr, np.identity(self.m_nisotopepairs))

        elif (atr == 'corr_mp_mp'):
            if np.all(self.m_ap == self.m_ap[0]):
                setattr(self, atr,
                        1 / (self.m_nisotopepairs * self.m_nisotopepairs)
                        * np.ones((self.m_nisotopepairs, self.m_nisotopepairs)))
            else:
                setattr(self, atr, np.identity(self.m_nisotopepairs))

        elif (atr == 'corr_m_mp'):
            setattr(self, atr, np.zeros((self.m_nisotopepairs,
                self.m_nisotopepairs)))

        elif (atr == 'corr_X_X'):
            setattr(self, atr, np.identity(self.n_ntransitions))

    def _init_elem(self):
        for (i, file_type) in enumerate(self.INPUT_FILES):
            if len(self.INPUT_FILES) != len(self.elem_init_atr):
                raise NameError("Number of INPUT_FILES does not match number of elem_init_atr.")

            file_name = file_type + self.id + '.dat'
            file_path = os.path.join(_data_path, self.id, file_name)

            # if not os.path.exists(file_path):
                # raise ImportError(f"Path {file_path} does not exist.")
            self.__load(self.elem_init_atr[i], file_type, file_path)

    def _init_corr_mats(self):

        corr_mats_to_be_defined = []

        for i, file_type in enumerate(self.OPTIONAL_INPUT_FILES):
            if len(self.OPTIONAL_INPUT_FILES) != len(self.elem_corr_mats):
                raise NameError("Number of OPTIONAL_INPUT_FILES does not match number of elem_corr_mats.")

            file_name = file_type + self.id + '.dat'
            file_path = os.path.join(_data_path, self.id, file_name)

            if os.path.exists(file_path):
                print("Loading {} for Element {}".format(
                    self.elem_corr_mats[i], self.id))
                self.__load(self.elem_corr_mats[i], file_type, file_path)
            else:
                self.__set_id_corr_mats(self.elem_corr_mats[i])
                corr_mats_to_be_defined.append(self.elem_corr_mats[i])

        if (len(corr_mats_to_be_defined) > 0):
            print("Using default values for {} of Element {}.".format(
                ', '.join(str(cm) for cm in corr_mats_to_be_defined), self.id))

    def _init_Xcoeffs(self):
        """
        Initialise the X coefficients to the set computed for a given mediator mass.

        """
        self.X = self.Xcoeffs[0]
        self.sig_X = self.sig_Xcoeffs[0]

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

        self.X = self.Xcoeffs[x]
        self.sig_X = self.sig_Xcoeffs[x]

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
            raise AttributeError("Passed fit parameters do not have appropriate dimensions")
        # assert (len(thetas[0]) == len(thetas[1]) == self.n_ntransitions - 1), (len(thetas[0]), self.n_ntransitions - 1)

        if np.array([(-np.pi / 2 > phij) or (phij > np.pi / 2) for phij in thetas[1]]).any():
            raise ValueError("Passed phij values are not within 1st / 4th quadrant.")

        # assert np.array([(-np.pi / 2 < phij) & (phij < np.pi / 2) for phij in
            # thetas[1]]).all()

        self.Kperp1 = np.insert(thetas[0], 0, 0.)
        self.ph1 = thetas[1]
        self.alphaNP = thetas[2]

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
        Generate uncertainties on mass normalised isotope shifts and write mxn-matrix to file.

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

           F1 = (1, tan(phi_12), ... , tan(phi_1n))

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
        # print("inside d_ai: this is F1", self.F1)
        # print("F1[1] type", self.F1[1].dtype)
        # print("inside d_ai: this is secph1", self.secph1)
        # print("secph1[1] type", self.secph1[1].dtype)
        # print("inside d_ai: this is Kperp1", self.Kperp1)
        # print("Kperp1[1] type", self.Kperp1[1].dtype)

        # print("inside d_ai: this is D_a1i(0,0)", self.D_a1i(0, 0))
        # print("inside d_ai: this is mu_aap", self.mu_aap)
        # print("inside d_ai: this is F1sq", self.F1sq)

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

    @cached_fct
    def fDdDnu_ij(self, i: int, j: int):
        """
        Return part of derivative of nu_i^a wrt. nu_j^b that does not depend on.
        isotope indices a = AA', b = BB'.

        """
        if ((i == 0) & (j == 0)):
            return (1 / self.F1sq * np.sum(np.array([self.F1[j]**2
                for j in self.range_j])))

        elif ((i == 0) & (j in self.range_j)):
            return -1 / self.F1sq * self.F1[j]

        elif ((i in self.range_j) & (j == 0)):
            return -1 / self.F1sq * self.F1[i]

        elif ((i in self.range_j) & (j in self.range_j)):
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
        fmat = np.array([[self.fDdDnu_ij(i, j) if i<=j else 0 for j in self.range_i]
            for i in self.range_i])
        fmat = fmat + fmat.T - np.diag(np.diag(fmat))

        return np.einsum('ab,ij->aibj', np.linalg.inv(np.diag(self.mu_aap)), fmat)

    @cached_fct
    def fDdDmmp_aib(self, a: int, i: int, b: int, merkki: int):

        if (i == 0):
            return (-merkki / self.F1sq / self.mu_aap[a]**2  # (self.m_a[b] * self.mu_aap[a])**2
                * np.sum(np.array([self.F1[j] * self.D_a1i(a, j) for j in self.range_j])))

        elif (i in self.range_j):
            return (merkki / self.mu_aap[a]**2  # (self.m_a[b] * self.mu_aap[a])**2 * (
                    * (self.D_a1i(a, i) - self.F1[i] / self.F1sq
                        * np.sum(np.array([self.F1[j] * self.D_a1i(a, j) for j in
                        self.range_j]))))

    @cached_fct
    def fDdDm_aib(self, a: int, i: int, b: int):
        """
        Return derivative of nu_i^a wrt. m^B, where a = AA' and B is a.
        reference isotope index.

        """
        if (a not in self.range_a):
            raise IndexError(f"Isotope pair index {a} passed to fDdDm_aib is out of range.")

        if (b not in self.range_a):
            raise IndexError(f"Isotope pair index {a} passed t fDdDm_aib is out of range.")

        # assert ((a in self.range_a) & (b in self.range_a)), (a, b)

        if (self.a_nisotope[b] == self.a_nisotope[a]):
            # print("(a,b, i)", (a, b, i))
            return self.fDdDmmp_aib(a, i, b, 1) / self.m_a[b]**2

        elif (self.a_nisotope[b] == self.ap_nisotope[a]):
            # print("(ap, b)", (a, b))
            return self.fDdDmmp_aib(a, i, b, -1) / self.m_a[b]**2

        else:
            return 0

    @cached_fct
    def fDdDmp_aib(self, a: int, i: int, b: int):
        """
        Return derivative of nu_i^a wrt. m^{B'}, where a = AA' and B' is a.
        primed isotope index.

        """
        if (a not in self.range_a):
            raise IndexError(f"Isotope pair index {a} is out of range.")

        if (b not in self.range_a):
            raise IndexError(f"Isotope pair index {a} is out of range.")

        if (self.ap_nisotope[b] == self.ap_nisotope[a]):
            return self.fDdDmmp_aib(a, i, b, -1) / self.m_ap[b]**2

        elif (self.ap_nisotope[b] == self.a_nisotope[a]):
            return self.fDdDmmp_aib(a, i, b, 1) / self.m_ap[b]**2

        else:
            return 0

    @cached_fct_property
    def DdDm(self):
        """
        Return derivative of d wrt. m, where m is the vector of reference
        isotope masses.

        """
        return np.array([[[self.fDdDm_aib(a, i, b) for b in self.range_a] for i
            in self.range_i] for a in self.range_a])

    @cached_fct_property
    def DdDmp(self):
        """
        Return derivative of d wrt. mp, where mp is the vector of primed
        isotope masses.

        """
        return np.array([[[self.fDdDmp_aib(a, i, b) for b in self.range_a] for i
            in self.range_i] for a in self.range_a])

    @cached_fct
    def DdDX_aij(self, a: int, i: int, j: int):
        """
        Return derivative of nu_i^a wrt. X_j, where a = AA' and j is a
        transition index.

        """
        if ((a in self.range_a) & (i == 0) & (j == 0)):
            return (-self.alphaNP / self.F1sq * self.h_aap[a]
                    * np.sum(np.array([self.F1[j]**2 for j in self.range_j])))

        elif ((a in self.range_a) & (i == 0) & (j in self.range_j)):
            return self.alphaNP / self.F1sq * self.h_aap[a] * self.F1[j]

        elif ((a in self.range_a) & (i in self.range_j) & (j == 0)):
            return (self.alphaNP * self.h_aap[a] * self.F1[i] * (1 - 1 / self.F1sq
                * np.sum(np.array([self.F1[k]**2 for k in self.range_j]))))

        elif ((a in self.range_a) & (i in self.range_j) & (j in self.range_j)):
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
        return np.array([[[self.DdDX_aij(a, i, j) for j in self.range_i] for i in
            self.range_i] for a in self.range_a])

    @cached_fct_property
    def cov_nu_nu(self):
        """
        Return covariance matrix of the isotope shift measurements nu, given
        the uncertainties on the isotope shift measurements and their
        correlations.

        """
        return np.einsum('ai,aibj,bj->aibj',
                self.sig_nu, self.corr_nu_nu, self.sig_nu)

    @cached_fct_property
    def cov_m_m(self):
        """
        Return covariance matrix of the reference isotope mass measurements.
        m_A, given the uncertainties on the isotope mass measurements and their
        correlations.

        """
        return np.einsum('a,ab,b->ab',
                self.sig_m_a, self.corr_m_m, self.sig_m_a)

    @cached_fct_property
    def cov_m_mp(self):
        """
        Return covariance matrix of the measurements of the reference isotope
        masses m_A and the primed isotope masses, given the experimental
        uncertainties and their correlations.

        """
        return np.einsum('a,ab,b->ab',
                self.sig_m_a, self.corr_m_mp, self.sig_m_ap)

    @cached_fct_property
    def cov_mp_mp(self):
        """
        Return covariance matrix of the primed isotope mass measurements
        m_Ap, given the experimental uncertainties and their correlations.

        """
        return np.einsum('a,ab,b->ab',
                self.sig_m_ap, self.corr_mp_mp, self.sig_m_ap)

    @cached_fct_property
    def cov_X_X(self):
        """
        Return covariance matrix of the X coefficients given the theoretical
        uncertainties and their correlations.

        """
        return np.einsum('i,ij,j->ij',
                self.sig_X, self.corr_X_X, self.sig_X)

    @cached_fct_property
    def cov_d_d(self):
        """
        Return the covariance matrix cov[d,d].

        """
        normald = self.dmat / self.absd[:, None]

        return (np.einsum('ai,aick,ckdl,bjdl,bj->ab',
                   normald, self.DdDnu, self.cov_nu_nu, self.DdDnu, normald)
               + np.einsum('ai,aic,cd,bjd,bj->ab',
                   normald, self.DdDm, self.cov_m_m, self.DdDm, normald)
               + np.einsum('ai,aic,cd,bjd,bj->ab',
                   normald, self.DdDm, self.cov_m_mp, self.DdDmp, normald)
               + np.einsum('ai,aic,cd,bjd,bj->ab',
                   normald, self.DdDmp, self.cov_m_mp.T, self.DdDm, normald)
               + np.einsum('ai,aic,cd,bjd,bj->ab',
                   normald, self.DdDmp, self.cov_mp_mp, self.DdDmp, normald)
               + np.einsum('ai,aik,kl,bjl,bj->ab',
                   normald, self.DdDX, self.cov_X_X, self.DdDX, normald))

    @cached_fct_property
    def LL(self):
        """
        Generate the contribution of the element to the negative log-likelihood LL.

        """
        return (1 / 2 * (self.absd @ np.linalg.inv(self.cov_d_d) @ self.absd))
