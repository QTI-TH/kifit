import os

from kifit.plot import multi_plot_mphi_alphaNP
from kifit.run import Runner
from kifit.config import RunParams

configurations_path = "./configurations/multifit_configs"

messengers_list = []

for conf in os.listdir(configurations_path):
    these_params = RunParams(configuration_file=f"{configurations_path}/{conf}")
    messengers_list.append(Runner(these_params))

multi_plot_mphi_alphaNP(messengers_list=messengers_list)