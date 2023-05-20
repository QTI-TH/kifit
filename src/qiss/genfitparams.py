import numpy as np
from functools import cache
from functools import cached_property
from qiss.loadelems import ElemData

import os


_data_path = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../data'
))


def sec(x: float):
    return 1 / np.cos(x)


class ElemFitParams:

    CACHE = {}

    def __init__(self, elem, x: int):
        self._elem = elem
        self.id = "fit_params_" + elem.id
        self._load_X_coeff_table()
        self._init_fit_params()
        self._init_X_coeffs(x)

    def _load_X_coeff_table(self):

        file_path = os.path.join(_data_path, self._elem.id, 'Xcoeffs' + self._elem.id + '.dat')
        print('loading table of X coefficients for element {} from {}'.format(self._elem.id, file_path))
        val = np.loadtxt(file_path)
        val = val.reshape(-1, self._elem.n_trans_nb)
        setattr(self, 'X_coeff_table', val)

    def _init_fit_params(self):

        self.Kperp1 = np.zeros(self._elem.n_trans_nb)
        self.ph1 = np.insert(np.zeros(self._elem.n_trans_nb - 1), 0, 1)
        self.alphaNP = 0

    def _init_X_coeffs(self, x: int):
        """
        Initialises the X coefficients to the set computed for a given mediator mass.

        """
        assert ((0 <= x) & (x <= len(self.X_coeff_table)))

        self.X_coeffs = self.X_coeff_table[x]

        assert len(self.X_coeffs) == self._elem.n_trans_nb, len(self.X_coeffs)

    def update_fit_params(self, thetas):
        """
        Sets the fit parameters

           thetas = {Kperp1, ph1, alphaNP},

        where Kperp1 and ph1 are (n-1)-vectors and alphaNP is a scalar, to the
        values provided in "thetas".

        """

        assert ((-np.pi / 2 < thetas[1]) & (thetas[1] < np.pi / 2)).all()

        self.Kperp1 = thetas[0]
        self.ph1 = thetas[1]
        self.alphaNP = thetas[2]

    @classmethod
    @cache
    def get(cls, elem: str, x: int):
        return cls(ElemData(elem), x)

    @classmethod
    def gen_all(cls, x: int):
        """
        Generates all fit parameters of all elements and returns result as dict
        """
        return {u: cls(ElemData(u), x) for u in ElemData.VALID_ELEM}

    def __repr__(self):
        return self.id + '[Kperp1, ph1, alphaNP]'

    @cached_property
    def np_term(self):
        """
        Generates the (m x n)-dimensional new physics term starting from
        theoretical input and fit parameters.

        """
        X1 = self.Xcoeffs[0] - np.tan(self.ph1) * self.Xcoeffs

        return self.alphaNP * np.multiply(X1, self._elem.h_np_nucl)

    @cached_property
    def F1(self):
        """
        Field shift vector entering King relation.

           F1 = (1, tan(phi_12), ... , tan(phi_1n))

        """
        return np.insert(np.tan(self.ph1), 0, 1)

    @cached_property
    def F1sq(self):
        """
        Squared norm of field shift vector F1.

        """
        return self.F1 @ self.F1

    @cached_property
    def range_a(self):
        """
        Returns range of isotope indices
           [0, 2, ...., m-1]
        """
        return range(self._elem.m_isopair_nb)

    @cached_property
    def range_i(self):
        """
        Returns range of transition indices
           [0, 2, ...., n-1]
        """
        return range(self._elem.n_trans_nb)

    @cached_property
    def range_j(self):
        """
        Returns range of indices of transitions that are not reference
        transitions.
           [1, ...., n-1]
        """
        return range(1, self._elem.n_trans_nb)

    def d_a_i(self, a: int, i: int):
        """
        Returns element d_i^{AA'} of the d-vector d^{AA'} in transition space.

        """

        if ((i == 0) & (a in self.range_a)):

            return - 1 / self.F1sq * np.sum(np.array([self.F1[j]
                * (self._elem.nu[a, j]
                    - sec(self.ph1[j]) * self.Kperp1[j]
                    - self.F1[j] * self._elem.nu[a, 0]
                    - self.np_term[a, j])
                for j in range(1, self._elem.n_trans_nb)]))

        elif ((i in self.range_i) & (a in self.range_a)):

            return (self._elem.nu[a, i]
                    - sec(self.ph1[i]) * self.Kperp1[i]
                    - self.F1[i] * self._elem.nu[a, 0]
                    - self.np_term[a, i]
                    + self.F1[i] * self.d_a_1(a))

        else:
            raise ValueError('transition index out of range.')

    def D_a_1i(self, a: int, i: int):
        """
        Returns object D_{1j}^a, where a is an isotope pair index and j is a
        transition index.

        """
        if ((i == 0) & (a in self.range_a)):

            return 0

        elif ((i in self.range_j) & (a in self.range_a)):
            return self._elem.nu[a, i] - self.ph1[i] * self._elem.nu[a, 1] - self.np_term[a, i]

        else:
            raise ValueError('index out of range.')

    def Ddai_Dnubj(self, a: int, i: int, b: int, j: int):
        """
        Returns derivative of nu_i^{AA'} wrt. nu_j^{BB'}

        """

        if ((i == 0) & (j == 0) & (a in self.range_a)):

            return np.sum(np.tan(self.ph1)**2) / self._elem.mu_invm[a] / self.F1sq

        elif ((((i == 0) & (j in self.range_j)) or ((i in self.range_j) & (j == 0))) & (a in self.range_a)):

            return -np.tan(self.ph1[j]) / self._elem.mu_invm[a] / self.F1sq

    @cached_property
    def cov_d(self):
        """
        Generates (m x m) x (n x n)-dimensional covariance matrix of the
        elements d_i^{AA'} of the m vectors in n-dimensional transition space.

        """

        # id_m_n = np.tensordot(np.identity(self.m),np.identity(self.n), axes=0)

        # cov_nu =
        # np.array([np.multiply(np.multiply(self.sig_nu[a,i], np.identity(self.n)), self.sig_nu[i])
        pass
        # Snu = [];
        # Snu[AAp,i] = (signu[AAp,i]/mu[AAp])**2;
        # Smu[AAp,BBp,i,j]= (mnu[AAp,i] * sigm[A]
        # self.reduced_isotope_shifts * reduced_isotope_shifts

    @cached_property
    def LL_elem(self, thetas):
        """
        Given the fit parameters

           thetas = {Kijperp, phiij, alphaNP},

        generates the contribution of the element to the log-likelihood LL.

        """
        pass


if __name__ == "__main__":
    ca = ElemData('Ca')
    np_term = ca.zeros()

    ca_efp = ElemFitParams(ca, 0)
    # ca_efp.d_a_1(0, np_term)
