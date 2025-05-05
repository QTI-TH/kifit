import os
import matplotlib.pyplot as plt

from kifit.plot import multi_plot_mphi_alphaNP
from kifit.run import Runner
from kifit.config import RunParams

plt.rc('text', usetex=True)
plt.rc('font', family='serif') 

configurations_path = "./configurations/calciums"

messengers_list = []

for conf in os.listdir(configurations_path):
    these_params = RunParams(configuration_file=f"{configurations_path}/{conf}")
    messengers_list.append(Runner(these_params))

multi_plot_mphi_alphaNP(
    messengers_list=messengers_list,
    # labels_list=["globalogrid", "detlogrid"],   
    # colors_list=["#e7a61b", "#7838c0", "darkgreen"],
    show_alg_for=[
            "Ca_WT_Aarhus_PTB_2024",
            # "Ca_PTB_2015",
            # "Ca24min",
            # "Yb_Kyoto_MIT_GSI_PTB_MPIK_2024",
        ],
    # algebraic_methods=["gkp", "nmgkp", "proj"],
    algebraic_methods=["gkp"],
    img_name="Calciums",
    dataset_name=True,
    # print_all_alg_results=True,
    # title=r"\texttt{detlogrid}"
    )