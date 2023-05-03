from abc import abstractmethod
import numpy as np
from pathlib import Path

import scipy.stats as sps
from scipy.optimize import minimize
import cma

def npmodel(par, data):
    """
    Model calculation, i.e. mv2 = k + F * mv1 + alpha X gamma

    Returns: all set of y-axis predictions
    """
    pass


def dummy_model(par, x):
    # THIS IS ONLY A DUMMY MODEL USED FOR TESTS
    return par[0]*x + par[1] + par[1]*np.log(x)


class Optimizer:
    """General optimizer class, which returns the best set of parameters."""

    _method = None

    def __init__(self, datapath):
        """
        Args: 
            datapath (str): path to target data folder.
        """
        
        # loading data
        self._data, self._g, self._x, self._npx, self._y = self.load_data(datapath)
        
        # dimensions and initial parameters setting
        self._dimensions = self._data.shape
        self._alpha = 0
        self._params = np.zeros((self._dimensions[1] - 1, 2))


    def load_data(self, datapath):
        """Loads data to be used during the optimization."""

        path = Path(datapath)

        data =  np.load(path/"nu.dat") 
        g = np.load(path/"g.dat") 
        npx = np.load(path/"npx.dat")    
        x = self._data.T[0]
        y = self._data.T[1:]

        return data, g, x, npx, y
    
    
    def set_reference_index(self, index):
        """Sets the new target index used as mv1 in each scatterplot"""

        self._x = self._data.T[index]
        self._y = np.delete(self._data, index, axis=1)
        # TO FIX:  what about g?

    
    def linear_fit(self):
        """updates linear fitting parameters."""

        lin_params = []
        for y in self._y:
            lin_params.append(sps.linregress(self._x, y))

        self._params = np.asarray(lin_params)



    def loss(self, data, params):
        """ Calculate loss function."""
        # THE FOLLOWING LOSS IS ONLY FOR TESTING
        
        

        for i, y in enumerate(self._y):
            prediction = 

        return 
    

    @abstractmethod
    def set_options(self, **kwargs):
        """Cast the options passed into the format expected by the optimizer"""
        pass


    def optimize(self, initial_p):
        """
        Optimization method which must be implemented according to each optimizer.
        
        Return: tuple [best loss function value, best parameters, full optimizer
        results dictionary].  
        """
        if self._method is None:
            raise ValueError(
                f"The optimizer {self.__class__.__name__} does not implement any methods"
            )
        pass

    

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


    def optimize(self, initial_parameters):
        """Calls the cma optimizer"""
        
        res = cma.fmin2(
            loss=self.loss,
            initial_parameters=initial_parameters,
            options=self._options,
            args=()
        )
        
        return res[1].result.fbest, res[1].result.xbest, res    