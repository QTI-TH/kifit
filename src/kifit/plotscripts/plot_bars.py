import json
import os
import copy

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import seaborn as sns

base_colors = ['#2ca02c', '#ff7f0e', '#9467bd']
def create_custom_palette(n):
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", base_colors, N=n)
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]


def draw_point(x, y, lb, ub, col, lab):
    plt.scatter(x, y, s=30, color=col, label=lab)
    plt.hlines(y, lb, ub, color=col)


def draw_set(alphas, lbs, ubs, title, lab, lab_array, keyword):
    # some decoration
    colors = create_custom_palette(len(lab_array))
    
    # some ylabel stuff
    xticks = [None]
    for i, l in enumerate(lab_array):
        xticks.append(str(l))
    xticks.append(None)
    
    plt.figure(figsize=(10 * 0.5, 10 * 0.5 * 6/8))
    for i in range(len(alphas)):
        draw_point(alphas[i], (i+1)*3, lbs[i], ubs[i], colors[i], lab+str(lab_array[i]))
    plt.title(title)
    plt.xlabel(r"$\alpha_{\rm NP}/\alpha_{\rm EM}$")
    plt.ylabel(r"Parameter value")
    plt.yticks(np.arange(0,len(lab_array)*3+4,3), xticks)
    plt.vlines(0, 0, len(lab_array)*3+2, color="black", ls="-", lw=1)
    plt.savefig(f"{keyword}.png", dpi=1000, bbox_inches="tight")


def load_data_using_keyword(path, keyword, ns):
    alphas, lbs, ubs = [], [], []

    params_dict = copy.deepcopy(PARAMS_DICT)

    for value in ns:
        for i in range(len(keyword)):
            params_dict.update({f"{keyword[i]}": value})
        json_filename = costruct_filename(**params_dict)
        with open(f"{path}/{json_filename}") as file:
            res_dict = json.load(file) 
        alphas.append(res_dict["best_alpha"])
        lbs.append(res_dict["LB"])
        ubs.append(res_dict["UB"])
    
    return alphas, lbs, ubs


def costruct_filename(
        data_name, 
        es_search,
        as_search,  
        logridfrac,
        exps, 
        es_exp, 
        as_exp, 
        minperc, 
        blocksize, 
        x0
    ):
    json_filename = (
        f"{data_name}_"
        + f"{es_search}es-search_"
        + f"{as_search}as-search_"
        + f"{logridfrac}logridfrac_"
        + f"{exps}exps_"
        + f"{es_exp}es-exp_"
        + f"{as_exp}as-exp_"
        + f"{minperc}minperc_"
        + f"blocksize{blocksize}_"
        + "sampling_fitparams_"
        + f"x{x0}.json"
    )
    return json_filename


path = "../results/output_data"
keyword = ["es_search", "es_exp"]
ns = [50, 100, 200, 500, 1000]
# ns = [10, 50, 100, 200, 500, 1000]
# ns = [-2,-5,-8,-10]
# ns = [50, 100, 200, 500, 1000]
# ns = ["TNC", "differential_evolution"]
# ns = np.arange(20,43,1)
# ns = [1,2,5,10,20]

PARAMS_DICT = {
    "data_name": "Camin",
    "es_search": 500,
    "as_search": 500,
    "logridfrac": -5,
    "exps": 50,
    "es_exp": 500,
    "as_exp": 500,
    "minperc": 5,
    "blocksize": 5,
    "x0": 0,
}


data = load_data_using_keyword(path, keyword, ns)
draw_set(data[0], data[1], data[2], r"Sampled elements", r"$n=$", ns, keyword)
