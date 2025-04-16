import json
from kifit.run import Runner
from kifit.config import RunParams
from kifit.plot import plot_bars

# Load the base configuration
config_file = "./configurations/base_config.json"
with open(config_file, 'r') as f:
    initial_config = json.load(f)  # Save initial configuration in a dictionary

# Initialize parameters and runner
params = RunParams(configuration_file=config_file)
runner = Runner(params)

# Explored values for the plot
values = [50, 100, 200, 500, 1000, 2000]
# values = [40, 200, 400, 800, 1200]
# values = [1, 2, 3, 5, 8]
plot_bars(runner, "elemsamples", values, r"Sampled elements")

# Restore the original configuration
with open(config_file, 'w') as f:
    json.dump(initial_config, f, indent=4)
