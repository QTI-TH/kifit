import numpy as np

import scipy.stats as sps
from scipy.optimize import minimize
import cma


def loss_function(parameters, collection):
    return collection.LL(parameters)


class Optimizer:
    """General optimizer class, which returns the best set of parameters."""

    def __init__(
        self,
        target_loss: float,
        max_iterations: int,
        bounds=[None, None],
        verbose: int = 1,
    ):
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
        self.bounds = bounds
        self.initial_params = None

    def optimize(self):
        """
        Perform the optimization strategy according to the chose method.

        Returns:
            Best loss function value registered.
            Best set of parameters.
            OptimizerResult object which depends on the used method.
        """
        raise NotImplementedError("Subclasses must implement the optimize() method.")

    def update_parameters(self, parameters):
        """Update initial parameters of the optimization."""
        self.initial_params = parameters

    def set_bounds(self, bounds):
        """Set a new bounds list."""
        self.bounds = bounds

    def get_linear_fit_params(self, data, reference_transition_idx: int = 0):
        """
        Perform linear regression.

        Args:
            data (normalised isotope shifts: rows=isotope pairs, columns=trans.)
            reference_transition_index (default: first transition)

        Returns:
            slopes, intercepts, Kperp, phi

        """

        x = data.T[reference_transition_idx]
        y = np.delete(data, reference_transition_idx, axis=1)

        slopes = []
        intercepts = []

        for i in range(y.shape[1]):
            results = sps.linregress(x, y.T[i])
            slopes.append(results.slope)
            intercepts.append(results.intercept)

        angles = np.arctan(slopes)
        return slopes, intercepts, intercepts * (np.cos(angles)), angles.tolist()


class CMA(Optimizer):
    """
    Call a CMA-ES optimizer.
    Ref: https://arxiv.org/abs/1604.00772
    """

    def __init__(
        self,
        target_loss,
        max_iterations,
        sigma0=1e-13,
        maxfeval=None,
        bounds=[None, None],
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
        super().__init__(target_loss, max_iterations, bounds, verbose)

        self._method = "cma"
        self.maxfeval = maxfeval
        self.sigma0 = sigma0

    def optimize(
        self, loss: callable, initial_parameters, args=()
    ) -> tuple[float, list]:
        """Call the CMA-ES optimizer."""

        options = {
            "verbose": self.verbose,
            "tolfun": self.target_loss,
            "maxiter": self.max_iterations,
            "maxfeval": self.maxfeval,
            "tolfunhist": 1e-16,
            "tolx": 1e-16,
            "bounds": self.bounds,
        }

        res = cma.fmin2(
            loss,
            initial_parameters,
            self.sigma0,
            args=[args],
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
        self.method = method
        self.jac = jac
        self.hess = hess
        self.hessp = hessp
        self.bounds = bounds
        self.costraints = costraints
        self.callback = callback
        self.options = options

    def optimize(self, loss, initial_parameters, args=()):
        """Execute the minimization according to the chosen method."""
        res = minimize(
            fun=loss,
            x0=initial_parameters,
            args=args,
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
        return res.fun, res.x  # , resa
