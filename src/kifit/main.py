from kifit.run import Runner
from kifit.config import RunParams


# set the kifit parameters from parser
params = RunParams()

# initialize the runner
runner = Runner(params)

# run
runner.run()

# runner.generate_all_alphaNP_ll_plots()
