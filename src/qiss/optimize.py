from abc import abstractmethod
import numpy as np

import scipy.stats as sps
from scipy.optimize import minimize
import cma

def npmodel(par, data):
    """
    Model calculation, i.e. mv2 = k + F * mv1 + alpha X gamma
    
    Args:
        par (array):  parameters matrix containg all the problem's parameters.
        Each row corresponds to a scatterplot and contains slope, intercept, and
        new physics coefficient for a given couple of isotopes.
        data (matrix): each raw represents a set of target x-axis data. 
    
    Returns: all set of y-axis predictions
    """

    return par[0] + par[1] * data + par[2]



class Optimizer:
    """General optimizer class, which returns the best set of parameters."""

    _method = None

    def __init__(self, datapath):
        """
        Args: 
            datapath (str): path to target data.
        """

        data =  np.load(datapath)

        # first matrix column as fixed abscissa
        self._x = data.T[0]
        # other columns as y variables
        self._y = data.T[1:]

        self.lin_params 

        # optimizer options
        self._options = {}


    def linear_fit(self):
        """
        Linear fit without new physics.
        
        Returns: list of intercepts and slopes associated to each mv2 mv1 combo.
        """

        lin_params = []

        for y in self._y:
            lin_params.append(sps.linregress(self._x, y))

        return lin_params


    def loss(self):
        """ Calculate loss function."""

        # sum over MSE residual \forall isotopes couples
        return 
    
    @abstractmethod
    def set_options(self, **kwargs):
        """Cast the options passed into the format expected by the optimizer"""
        pass


    def optimize(self, initial_p):
        if self._method is None:
            raise ValueError(
                f"The optimizer {self.__class__.__name__} does not implement any methods"
            )
        return minimize(self.loss, initial_p, method=self._method, options=self._options)
    

class CMA(Optimizer):
    """
    Calls a CMA-ES optimizer. 
    Ref: https://arxiv.org/abs/1604.00772 
    """
    _method = "cma"

    def set_options(self, **kwargs):
        self._options = {
            "verbose": -1,
            "tolfun": 1e-12,
            "ftarget": kwargs["tol_error"],  # target error
            "maxiter": kwargs["max_iterations"],  # maximum number of iterations
            "maxfeval": kwargs["max_evals"],  # maximum number of function evaluations
        }
        

    def optimize(self, initial_p):
        """Calls the cma optimizer"""

        # TO DO
    


class BFGS(Optimizer):
    """
    Calls the scipy's Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimizer.
    """

    _method = "bfgs"

    def set_options(self, **kwargs):
        self._options = {"disp": True, "return_all": True}
        print(f"Initial parameters: {self._predictor.parameters}")
    