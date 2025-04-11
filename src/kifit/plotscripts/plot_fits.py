import os

from kifit.plot import multi_plot_mphi_alphaNP
from kifit.run import Runner
from kifit.config import RunParams

configurations_path = "./configurations/combo"

messengers_list = []

for conf in os.listdir(configurations_path):
    these_params = RunParams(configuration_file=f"{configurations_path}/{conf}")
    messengers_list.append(Runner(these_params))

multi_plot_mphi_alphaNP(
    messengers_list=messengers_list,
    show_determinant_for=[
            "Ca_WT_Aarhus_PTB_2024",
            # "Ca_PTB_2015",
            # "Ca24min",
            # "Yb_Kyoto_MIT_GSI_PTB_MPIK_2024",
        ],
    img_name="Combo",
    )