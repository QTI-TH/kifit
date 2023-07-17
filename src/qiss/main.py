from optimize import CMA
from loadelems import Elem
from plots import plot_linear_fits

ca = Elem("Ca")
opt = CMA(target_loss=1e-6, max_iterations=100)

data = ca.mu_norm_isotope_shifts
print(f"Test data used to search for NP:\n{data}")

slopes, intercepts = opt.get_linear_fit_params(data, reference_transition_idx=0)

print("Plotting linear fits")
plot_linear_fits(slopes=slopes, intercepts=intercepts, data=data, target_index=0)

# initialize alphaNP
alphaNP = 0

parameters = []
parameters.append(slopes)
parameters.append(intercepts)
parameters.append(alphaNP)

print(ca.loss_function(parameters))
