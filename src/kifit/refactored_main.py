from kifit.run import Runner
from kifit.config import RunParams


# set the kifit parameters from parser
params = RunParams()

# initialize the runner
runner = Runner(params)

# print the element list
print(runner.config.params.element_list)

# save this config into a file
runner.dump_config("dumped_config.json")

# update the runner with a previously dumped configuration
runner.load_config("test_params1.json")

# print the new element list
print(runner.config.params.element_list)

# reload the initial configuration 
runner.load_config("dumped_config.json")

# run
runner.run()




