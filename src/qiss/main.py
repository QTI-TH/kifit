from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from qiss.optimizers import CMA, loss_function
from qiss.loadelems import ElemCollection


# parsing arguments
parser = ArgumentParser()
parser.add_argument(
    "--nruns",
    help="Number of times the optimization has to be repeated",
    type=int,
    default=1,
)
parser.add_argument(
    "--datapath", help="Data folder path", type=Path, default="../qiss_data/"
)
parser.add_argument(
    "--output_folder",
    help="Output folder to save data",
    type=Path,
    default="../qiss/output/",
)
args = parser.parse_args()
print("datapath", args.datapath)

# define the element collection
collection = ElemCollection()
collection.init_collection(args.datapath)

# get collection parameters
params = collection.get_parameters
print("params", params)

# Optimizer initialization
sigma0 = 1e-13
opt = CMA(target_loss=-100, max_iterations=500, sigma0=1e-12, verbose=-1)

# define bounds for the optimization
delta = 1e-4  # 1e-3  # runs for 1e-3
low_bounds = np.asarray(params) - delta
upp_bounds = np.asarray(params) + delta

opt.set_bounds([low_bounds, upp_bounds])

print("Optimization starts!")

parameter_estimates = []
for i in range(args.nruns):
    res = opt.optimize(loss_function, initial_parameters=[params], args=collection)
    parameter_estimates.append(res[1])

print(f"alphaNP estimated: {res[1]}")

# np.save(file=args.output_folder / f"hist_{sigma0}", arr=np.asarray(parameter_estimates).T[-1])
