from optimize import CMA, loss_function
from loadelems import Elem
from plots import plot_linear_fits


elements = []

ca = Elem("Ca")

opt = CMA(target_loss=-100, max_iterations=500, bounds=[None, None])

elements.append(ca)
elements.append(ca)


data = ca.mu_norm_isotope_shifts
print(f"\nTest data used to search for NP:\n{data}")

slopes, intercepts, kperp1, ph1 = opt.get_linear_fit_params(
    data, reference_transition_idx=0
)

print("\nSave plots of linear fits")
plot_linear_fits(slopes=slopes, intercepts=intercepts, data=data, target_index=0)

# initialize alphaNP
alphaNP = 0
params = []
params.extend(kperp1)
params.extend(ph1)
params.extend(kperp1)
params.extend(ph1)
params.append(alphaNP)

print(params)

print(f"\nFancy loss function: {loss_function(parameters=params, elements=elements)}")

print(f"Start the optimization process")
res = opt.optimize(loss_function, initial_parameters=[params], args=[elements])
print(res[1])
