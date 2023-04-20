from abc import abstractmethod
import numpy as np
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

    return # TO BE IMPLEMENTED



class Optimizer:
    """
    Parent optimizer class. 
    It returns the best set of parameters.
    """

    _method = None

    def __init__(self, datapath):
        """
        Args: 
            datapath (str): path to target data.
        """

        _data =  np.load(datapath)
        _x = _data.T[0]
        
        # optimizer options
        _options = {}


    def linear_fit(self):
        """Linear fit without new physics"""

        # given i != 0
        i = 1
        params = np.polyfit(x=self._x, y=self._data.T[i], deg=1)
        return params


    def loss(self):
        """ Calculate loss function."""
        return 
    
    @abstractmethod
    def set_options(self, **kwargs):
        """Cast the options passed into the format expected by the optimizer"""
        pass
    

class CMA(Optimizer):
    
    def set_options(self, **kwargs):
        self._options = {
            "verbose": -1,
            "tolfun": 1e-12,
            "ftarget": kwargs["tol_error"],  # target error
            "maxiter": kwargs["max_iterations"],  # maximum number of iterations
            "maxfeval": kwargs["max_evals"],  # maximum number of function evaluations
        }