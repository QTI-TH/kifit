import numpy as np

import scipy.stats as sps
from scipy.optimize import minimize
import cma


class Optimizer:
    """General optimizer class, which returns the best set of parameters."""

    def __init__(self, target_loss: float, max_iterations: int, verbose: int = 1):
        """
        Args:
            datapath (str): path to target experiment folder.
                each experiment folder must be organised with one subfolder for
                each element involved into the Kings Plot fitting.
            target_loss (float): target loss function value at which the optimization
                can be stopped.
            max_iterations (int): maximum number of iterations.
            verbose (int): verbosity level increasing from -1 (quiet) to +1 (verbose).
        """

        self._method = None
        self.target_loss = target_loss
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.loss = self.build_loss_function()

    def optimize(self):
        """
        Perform the optimization strategy according to the chose method.

        Returns:
            Best loss function value registered.
            Best set of parameters.
            OptimizerResult object which depends on the used method.
        """
        raise NotImplementedError("Subclasses must implement the optimize() method.")

    def build_loss_function(self):
        """Build loss function using data of a target experiment."""
        pass

    def get_linear_fit_params(self, data, reference_transition_idx):
        """
        Perform linear regression.

        Args:
            data
            reference_transition_index
        """

        # indipendent variable
        x = data.T[reference_transition_idx]
        # data without the reference column used as indipendent
        y = np.delete(data, reference_transition_idx, axis=1)

        linear_params = []

        for i in range(y.shape[1]):
            linear_params.append(sps.linregress(x, y.T[i]))

        return linear_params


class CMA(Optimizer):
    """
    Call a CMA-ES optimizer.
    Ref: https://arxiv.org/abs/1604.00772
    """

    def __init__(
        self,
        target_loss,
        max_iterations,
        initial_params=None,
        maxfeval=None,
        verbose=1,
    ):
        """
        Args:
            datapath: path to target experiment folder.
                each experiment folder must be organised with one subfolder for
                each element involved into the Kings Plot fitting.
            target_loss: target loss function value at which the optimization
                can be stopped.
            max_iterations: maximum number of iterations.
            verbose: if True, log messages are printed during the optimization.
            initial_params: initial guess of paramameters.
            maxfeval: maximum number of function evaluations.
        """
        super().__init__(target_loss, max_iterations, verbose)

        self._method = "cma"
        self.initial_params = initial_params
        self.maxfeval = maxfeval

    def optimize(self) -> tuple[float, list]:
        """Call the CMA-ES optimizer."""

        options = {
            "verbose": self.verbose,
            "tolfun": self.target_loss,
            "maxiter": self.max_iterations,
            "maxfeval": self._maxfeval,
        }

        res = cma.fmin2(
            loss=self.loss,
            initial_parameters=self.initial_params,
            args=(),
            options=options,
        )

        return res[1].result.fbest, res[1].result.xbest, res


class ScipyMinimizer(Optimizer):
    """
    Call scipy.optimize.minimize method.
    Official documentation at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    def __init__(
        self,
        target_loss: float,
        max_iterations: int,
        initial_params: list,
        method: str = None,
        jac: callable = None,
        hess: callable = None,
        hessp: callable = None,
        bounds: tuple = None,
        costraints: dict = None,
        callback: callable = None,
        options: dict = None,
        verbose: int = 1,
    ):
        """
        Args:
            datapath: path to target experiment folder.
                each experiment folder must be organised with one subfolder for
                each element involved into the Kings Plot fitting.
            target_loss: target loss function value at which the optimization
                can be stopped.
            max_iterations: maximum number of iterations.
            verbose: if True, log messages are printed during the optimization.
            initial_params: initial guess of paramameters.
            maxfeval: maximum number of function evaluations.
            method: name of method supported by ``scipy.optimize.minimize``. If not given,
                chosen to be one of BFGS, L-BFGS-B, SLSQP, depending on whether
                or not the problem has constraints or bounds.
            jac: method for computing the gradient vector for scipy optimizers.
            hess: method for computing the hessian matrix for scipy optimizers.
            hessp: hessian of objective function times an arbitrary
                vector for scipy optimizers.
            bounds: bounds on variables for scipy optimizers.
            constraints: constraints definition for scipy optimizers.
            callback: called after each iteration for scipy optimizers.
            options: dictionary with options accepted by ``scipy.optimize.minimize``.
        """
        super().__init__(target_loss, max_iterations, verbose)

        self._method = f"scipy_optimizer_{method}"
        self.initial_params = initial_params
        self.method = method
        self.jac = jac
        self.hess = hess
        self.hessp = hessp
        self.bounds = bounds
        self.costraints = costraints
        self.callback = callback
        self.options = options

    def optimize(self):
        """Execute the minimization according to the chosen method."""
        res = minimize(
            fun=self.loss,
            x0=self.initial_params,
            args=(),
            method=self.method,
            jac=self.jac,
            hess=self.hess,
            hessp=self.hessp,
            bounds=self.bounds,
            constraints=self.costraints,
            tol=self.target_loss,
            callback=self.callback,
            options=self.options,
        )
        return res.fun, res.x, res
