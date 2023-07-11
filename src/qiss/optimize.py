from abc import abstractmethod
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import os

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


@dataclass
class Element:
    """Each element should be associated to an Element class"""
    name: str
    ndata: int
    x: np.ndarray
    y: np.ndarray
    g: np.ndarray
    npx: np.ndarray
    parameters: np.ndarray


class Optimizer:
    """General optimizer class, which returns the best set of parameters."""

    _method = None

    def __init__(self, datapath):
        """
        Args: 
            datapath (str): path to target experiment folder.
                each experiment folder must be organised with one subfolder for
                each element involved into the Kings Plot fitting.
        """
        
        # parent path to the experiment folder
        self._path = Path(datapath)
        self._elements = []

        for element_name in os.listdir(self._path):
            # loading data from element folder
            data, g, x, npx, y = self.load_data(element_name)
            # appending to _elements list the new element class
            self._elements.append(Element(
                name=element_name,
                ndata=data.shape[0],
                x=x,
                y=y,
                g=g,
                npx=npx
            ))


    def load_data(self, element_name):
        """Load data to be used during the optimization."""

        element_path = self._path/element_name

        data =  np.load(element_path/"nu.dat") 
        g = np.load(element_path/"g.dat") 
        npx = np.load(element_path/"npx.dat")    
        
        x = self._data.T[0]
        y = self._data.T[1:]

        return data, g, x, npx, y
    
    
    def set_reference_index(self, index):
        """Set the new target index used as mv1 in each scatterplot"""

        self._x = self._data.T[index]
        self._y = np.delete(self._data, index, axis=1)
        # TO FIX:  what about g?

    
    def linear_fit(self):
        """Update linear fitting parameters."""

        lin_params = []
        for y in self._y:
            lin_params.append(sps.linregress(self._x, y))

        self._params = np.asarray(lin_params)



    def loss(self, data, params):
        """ Calculate loss function."""

        for elem in self._elements:
            #use elem.data for calculating the loss function!
        pass
    

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
    Call a CMA-ES optimizer. 
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