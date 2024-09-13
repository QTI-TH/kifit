import json 

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#0067a5"
RED = "#cc3333"
GREEN = "#0cab1e"
PURPLE = "#7840bd"
ORANGE = "#f0a12b"

WIDTH = 0.5

elemlist = ["Ca_WT_Aarhus_PTB_2024"]
elnames = ["Ca", "Yb"]
elhatches = ["//", "\\\\"]
elcolors = [PURPLE, RED]
LIGHT_COLS = ["#f2acf2", "#faeda2"]

mphis = np.loadtxt("../../kifit_data/Ca_WT_Aarhus_PTB_2024/Xcoeffs_Ca_WT_Aarhus_PTB_2024.dat")
mphis = np.array(mphis).T[0]

gkps, projs = [], []

if len(elemlist) > 1:
    elements = "Yb_Kyoto_MIT_GSI_PTB_MPIK_2024_Ca_WT_Aarhus_PTB_2024"
else:
    elements = elemlist[0]

for i in range(len(elemlist)):
    gkps.append(f"../results/output_data/{elemlist[i]}_3-dim_gkp_1000samples_x")
    projs.append(f"../results/output_data/{elemlist[i]}_3-dim_proj_1000samples_x")

name1 = f"../results/output_data/{elements}_500es-search_500as-search_-5logridfrac_50exps_500es-exp_500as-exp_5minperc_blocksize10_sampling_fitparams_x"
# name2 = f"../results/output_data/{elements}_Powell_20searches_400es-search_50exps_400es-exp_400as-exp_5minperc_maxiter1000_blocksize5_globalopt_x"
name2 = f"../results/output_data/{elements}_500es-search_500as-search_-5logridfrac_50exps_500es-exp_500as-exp_5minperc_blocksize10_x"


ubs1, ubs2, lbs1, lbs2 = [], [], [], []
ubs_gkps, lbs_gkps, ubs_projs, lbs_projs = [], [], [], []

indexes = np.arange(0, 801, 25)
mphix = mphis[indexes]

for i in range(len(elemlist)):
    print(i)
    these_ubs_gkp, these_lbs_gkp, these_ubs_proj, these_lbs_proj = [], [], [], []

    for x in indexes:
        res_dict_gkps, res_dict_projs = [], []
        with open(f"{gkps[i]}{x}.json") as file:
                res_dict_gkp = json.load(file) 
        with open(f"{projs[i]}{x}.json") as file:
                res_dict_proj = json.load(file) 

        if i == 0:
            with open(f"{name1}{x}.json") as file:
                    res_dict1 = json.load(file)
            with open(f"{name2}{x}.json") as file:
                    res_dict2 = json.load(file) 
            ubs1.append(res_dict1["UB"])
            ubs2.append(res_dict2["UB"])
            lbs1.append(res_dict1["LB"])
            lbs2.append(res_dict2["LB"])
        these_ubs_gkp.append(res_dict_gkp["minpos"])
        these_lbs_gkp.append(res_dict_gkp["maxneg"])
        these_ubs_proj.append(res_dict_proj["minpos"])
        these_lbs_proj.append(res_dict_proj["maxneg"])

    ubs_gkps.append(these_ubs_gkp)
    lbs_gkps.append(these_lbs_gkp)
    ubs_projs.append(these_ubs_proj)
    lbs_projs.append(these_lbs_proj)

min_ub1 = np.nanmin(ubs1)
max_lb1 = np.nanmax(lbs1)

yticks = [-1e-1, -1e-6, -1e-10, 0,  1e-10, 1e-6, 1e-1]
linlim1 = 10 ** np.floor(np.log10(np.nanmax([np.abs(min_ub1), np.abs(max_lb1)])) - 1)


plt.figure(figsize=(10 * WIDTH, 10 * WIDTH * 6/8))
plt.plot(mphix, ubs1, color=ORANGE, lw=1.5, alpha=0.75, label=r"$2\sigma$ fit, alg. warmup")
plt.plot(mphix, lbs1, color=ORANGE, lw=1.5, alpha=0.75)
plt.plot(mphix, ubs2, color=GREEN, lw=1.5, alpha=0.75, label=r"$2\sigma$ fit, opt. warmup")
plt.plot(mphix, lbs2, color=GREEN, lw=1.5, alpha=0.75)

plt.fill_between(mphix, ubs1, 1e1, facecolor=ORANGE, alpha=0.3)  
plt.fill_between(mphix, -1e1, lbs1, facecolor=ORANGE, alpha=0.3)  

for i in range(len(elemlist)):
    plt.fill_between(mphix, ubs_gkps[i], 1e1, facecolor="none", edgecolor=elcolors[i], hatch=elhatches[i], lw=1.5)  
    plt.fill_between(mphix, -1e1, lbs_gkps[i], facecolor="none", edgecolor=elcolors[i], hatch=elhatches[i], lw=1.5)  
    plt.plot(mphix, ubs_gkps[i], color=elcolors[i], label=rf"$2\sigma$ dim-3 GKP {elnames[i]}", lw=1.5)
    plt.plot(mphix, lbs_gkps[i], color=elcolors[i], lw=1.5)
    plt.plot(mphix, ubs_projs[i], color=LIGHT_COLS[i], ls=":", label=rf"$2\sigma$ dim-3 proj {elnames[i]}", lw=1.5)
    plt.plot(mphix, lbs_projs[i], color=LIGHT_COLS[i], ls=":", lw=1.5)


plt.hlines(0, min(mphis[indexes]), max(mphis[indexes]), color="black", lw=1)
plt.hlines(linlim1, min(mphis[indexes]), max(mphis[indexes]), color="black", lw=1, ls="--")
plt.hlines(-linlim1, min(mphis[indexes]), max(mphis[indexes]), color="black", lw=1, ls="--")
plt.yscale("symlog", linthresh=linlim1)
plt.xscale("log")
plt.yticks(yticks)
plt.legend(fontsize=8, ncols=2, loc=2, framealpha=1)
plt.ylabel(r"$\alpha_{\rm NP}/\alpha_{\rm EM}$", fontsize=12)
plt.xlabel(r"m$_{\phi}$ [eV]", fontsize=12)
plt.title("Ca")

plt.savefig("ca_fits.pdf", bbox_inches="tight")