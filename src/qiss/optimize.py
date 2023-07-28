import numpy as np

import scipy.stats as sps
from scipy.optimize import minimize
from bayes_opt import BayesianOptimization
import cma


def loss_function(parameters, elements):
    """
    Build the loss function associated to a collection of elements.

    Args:
        parameters: flatten list of parameters containing Kperp1 and ph1 for all
            the elements in `elements`. The last list item must be alphaNP.
        elements: list of `loadelems.Elem` involved into the experiment.
    """

    # get alpha NP
    alphaNP = parameters[-1]
    # initialize index list to be zero
    parameter_index = 0
    # initial loss function value
    ll = 0

    # cycle over the parameters
    for elem in elements:
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

    def get_linear_fit_params(self, data, reference_transition_idx: int = 0):
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
        bounds,
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
        self.maxfeval = maxfeval
        self.bounds = bounds

    def optimize(
        self, loss: callable, initial_parameters, args=()
    ) -> tuple[float, list]:
        """Call the CMA-ES optimizer."""

        # params = []
        # for elem in initial_parameters:
        #     params.extend(elem[0])
        #     params.extend(elem[1])
        #     params.append(elem[2])

        # print(params)

        options = {
            "verbose": self.verbose,
            "tolfun": self.target_loss,
            "maxiter": self.max_iterations,
            "maxfeval": self.maxfeval,
            "bounds": self.bounds,
        }

        res = cma.fmin2(
            loss,
            initial_parameters,
            1e-3,
            args=args,
            options=options,
        )

        return res[1].result.fbest, res[1].result.xbest, res


class BayesianOptimizer(Optimizer):
    """
    Compute a Bayesian optimization using: https://github.com/bayesian-optimization/BayesianOptimization.
    """

    def __init__(
        self,
        target_loss,
        max_iterations,
        bounds=None,
        init_points=2,
        random_state=1,
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

        self._method = "bayesian"
        self.pbounds = bounds
        self.random_state = random_state
        self.init_points = init_points

    def optimize(self, loss, parameters, args=()):
        """Compute the optimization process."""
        optimizer = BayesianOptimization(
            f=loss, pbounds=self.pbounds, random_state=self.random_state
        )

        res = optimizer.maximize(
            init_points=self.init_points, n_iter=self.max_iterations
        )

        return res


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

    def optimize(self, loss):
        """Execute the minimization according to the chosen method."""
        res = minimize(
            fun=loss,
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
