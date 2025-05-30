import json
import os
import copy

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def draw_point(x, y, lb, ub, col, lab):
    plt.scatter(x, y, s=30, color=col, label=lab)
    plt.hlines(y, lb, ub, color=col)


def draw_set(alphas, lbs, ubs, title, lab, lab_array, keyword):
    # some decoration
    colors = sns.color_palette("inferno", n_colors=len(alphas)).as_hex()
    # some ylabel stuff
    xticks = [None]
    for i, l in enumerate(lab_array):
        xticks.append(str(l))
    xticks.append(None)

    plt.figure(figsize=(6, 6 * 6 / 8))
    for i in range(len(alphas)):
        draw_point(
            alphas[i], (i + 1) * 3, lbs[i], ubs[i], colors[i], lab + str(lab_array[i])
        )
    plt.title(title)
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$n$")
    plt.yticks(np.arange(0, len(lab_array) * 3 + 4, 3), xticks)
    plt.vlines(0, 0, len(lab_array) * 3 + 2, color="black", ls="-", lw=1)
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
    optimizer,
    searches,
    es_search,
    exps,
    es_exp,
    as_exp,
    minperc,
    maxiter,
    blocksize,
    globalopt,
    x0,
):
    json_filename = (
        f"{data_name}_"
        + f"{optimizer}_"
        + f"{searches}searches_"
        + f"{es_search}es-search_"
        + f"{exps}exps_"
        + f"{es_exp}es-exp_"
        + f"{as_exp}as-exp_"
        + f"{minperc}minperc_"
        + f"maxiter{maxiter}_"
        + f"blocksize{blocksize}_"
        + ("globalopt_" if globalopt else "")
        + f"x{x0}.json"
    )
    return json_filename


path = "../results/output_data"
keyword = ["minperc"]
# ns = [50, 100, 200, 500, 1000]
# ns = [50, 100, 200, 500, 1000]
# ns = [50, 100, 200, 500, 1000]
# ns = ["TNC", "differential_evolution"]
# ns = np.arange(20,43,1)
ns = [1, 2, 5, 10, 20]

PARAMS_DICT = {
    "data_name": "Ca_WT_Aarhus_PTB_2024",
    "optimizer": "Powell",
    "searches": 20,
    "es_search": 200,
    "exps": 20,
    "es_exp": 200,
    "as_exp": 200,
    "minperc": 5,
    "maxiter": 1000,
    "blocksize": 10,
    "globalopt": False,
    "x0": 400,
}


data = load_data_using_keyword(path, keyword, ns)
draw_set(data[0], data[1], data[2], r"Varying min percentile", r"$n=$", ns, keyword)
