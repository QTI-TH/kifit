from optimize import CMA
from loadelems import Elem
from plots import plot_linear_fits


elements = []

ca = Elem("Ca")
opt = CMA(target_loss=1e-6, max_iterations=100)

elements.append(ca)


def loss_function(params):
    """Loss function to be optimized"""  # TO DO --> GENERALIZE TO MORE ELEMENTS
    ca._update_fit_params(params)
    return -ca.LL


data = ca.mu_norm_isotope_shifts
print(f"\nTest data used to search for NP:\n{data}")

slopes, intercepts, kperp1, angles = opt.get_linear_fit_params(
    data, reference_transition_idx=0
)

print("\nSave plots of linear fits")
plot_linear_fits(slopes=slopes, intercepts=intercepts, data=data, target_index=0)

# initialize alphaNP
alphaNP = 0
params = []
params.extend(kperp1)
params.extend(angles)
params.append(alphaNP)

print(f"\nLoss function: {loss_function(params)}")

print(f"Start the optimization process")
opt.optimize(loss_function, initial_parameters=[params])
