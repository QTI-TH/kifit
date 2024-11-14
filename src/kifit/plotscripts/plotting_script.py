from kifit.run import Runner
from kifit.config import RunParams
from kifit.plot import plot_bars

# loading base config of the barplots experiments
params = RunParams(configuration_file="base_config.json")

runner = Runner(params)

# explored values
values = [100, 200, 500, 1000]

plot_bars(runner, "elemsamples", values)