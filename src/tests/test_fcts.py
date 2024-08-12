import numpy as np

# from tdqm import tqdm

from kifit.fitools import (generate_alphaNP_samples, compute_ll_experiments,
    get_delchisq, get_delchisq_crit, get_confint, get_bestalphaNP_and_bounds)


def get_new_alphaNP_interval(
        alphalist,
        llist,
        scalefactor: float = .1):

    smalll_inds = np.argsort(llist)[: int(len(llist) * scalefactor)]

    small_alphas = alphalist[smalll_inds]
    smallls = llist[smalll_inds]

    lb_best_alpha = np.percentile(small_alphas, 16)
    ub_best_alpha = np.percentile(small_alphas, 84)

    best_alpha = np.median(small_alphas)

    return (small_alphas, smallls,
        best_alpha, lb_best_alpha, ub_best_alpha)


def get_bestalpha_and_logL_spread_of_next_iteration(
        new_alphas,
        new_lls,
        current_alphas,
        current_lls,
        scalefactor: float = .1):

    new_best_alphas = new_alphas[new_lls < np.percentile(new_lls, 10)]

    new_best_alpha = np.median(new_best_alphas)
    sig_new_best_alpha = np.std(new_alphas)  # don't want to be too aggressive

    alphainds_best_window = np.where(np.logical_and(
        (min(new_best_alphas) < current_alphas), (current_alphas < max(new_best_alphas))
    ))[0]

    if len(alphainds_best_window) < 5:
        raise IndexError(f"""Increase number of samples.
        {len(alphainds_best_window)} points in new best window.""")

    Delta_new_ll = (max(current_lls[alphainds_best_window])
        - min(current_lls[alphainds_best_window]))

    return new_best_alpha, sig_new_best_alpha, Delta_new_ll


def update_alphaNP_for_next_iteration(
        elem,
        new_alphas,
        new_lls,
        alphalist,
        llist,
        scalefactor: float = .1):
    """
    Compute sig_alphaNP for next iteration.

    """
    new_alpha = np.median(new_alphas)
    std_new_alphas = np.std(new_alphas)

    new_best_alphas = new_alphas[new_lls < np.percentile(new_lls, 10)]

    alphainds_best_window = np.where(np.logical_and(
        (min(new_best_alphas) < alphalist), (alphalist < max(new_best_alphas))
    ))[0]

    if len(alphainds_best_window) < 5:
        raise IndexError(f"""Increase number of samples.
        {len(alphainds_best_window)} points in new best window.""")

    Delta_new_ll = max(llist[alphainds_best_window]) - min(llist[alphainds_best_window])

    sig_new_alpha = np.min([
        np.abs(max(new_alphas) - new_alpha),
        np.abs(min(new_alphas) - new_alpha)])

    elem.set_alphaNP_init(new_alpha, sig_new_alpha)

    print(f"""New search interval: \
        [{elem.alphaNP_init - elem.sig_alphaNP_init}, \
        {elem.alphaNP_init + elem.sig_alphaNP_init}]""")

    return Delta_new_ll, new_alpha, std_new_alphas, sig_new_alpha


def equilibrate_interval(newalphas, newlls,
        alpha_window_lfrac, alpha_window_ufrac):

    sorted_alpha_inds = np.argsort(newalphas)
    newalphas = newalphas[sorted_alpha_inds]
    newlls = newlls[sorted_alpha_inds]

    nnewpts = len(newlls)

    sorted_ll_inds = np.argsort(newlls)
    maxll_inds = sorted_ll_inds[: int(nnewpts / 10)]

    minll = min(newlls)
    min_llim = min(newlls[0], newlls[-1]) - minll
    max_llim = max(newlls[0], newlls[-1]) - minll

    llequill = min_llim / max_llim

    if (
        llequill < .4
        and np.all(
            (maxll_inds < nnewpts * alpha_window_lfrac)
            | (nnewpts * alpha_window_ufrac < maxll_inds))
    ):
        limiting_ll = min(newlls[0], newlls[-1])
        # print("newlls     ", newlls)
        # print("min ll     ", min(newlls))
        # print("max ll     ", max(newlls))
        # print("limiting_ll", limiting_ll)
        reduced_ll_inds = np.where(newlls <= limiting_ll)
        newalphas = newalphas[reduced_ll_inds]
        newlls = newlls[reduced_ll_inds]
    return newalphas, newlls


def iterative_mc_search(
        elem,
        nelemsamples_search: int = 200,
        nalphasamples_search: int = 200,
        nexps: int = 1000,
        nelemsamples_exp: int = 1000,
        nalphasamples_exp: int = 1000,
        nsigmas: int = 2,
        nblocks: int = 10,
        sigalphainit: float = 1.,
        scalefactor: float = .1,  # 2e-1,
        # sig_new_alpha_fraction: float = 0.1,
        maxiter: int = 1000,
        plot_output: bool = False,
        xind=0,
        mphivar: bool = False):

    """
    Perform iterative search for best alphaNP value and the standard deviation.

    Args:
        elem (Elem): target element.
        nsamples_search (int): number of samples.
        nsigmas (int): confidence level in standard deviations for which the
            upper and lower bounds are computed.
        nblocks (int): number of blocks used for the determination of the mean
            and standard deviation in the last iteration.
        scalefactor (float): factor used to rescale the search interval.
        maxiter (int): maximum number of iterations spent within the iterative
            search.

    Return:
        mean_best_alpha: best alphaNP value, found by averaging over all blocks
        sig_best_alpha: standard deviation of the best_alphas over all blocks
        LB: lower nsigma-bound on alphaNP
        sig_LB: uncertainty on LB
        UB: upper nsigma-bound on alphaNP
        sig_UB: uncertainty on UB

    """
    if plot_output:
        from kifit.plotfit import plot_mc_output, plot_final_mc_output

    alphas_exps = []
    lls_exps = []
    bestalphas_exps = []

    # if mphivar is False:
    #     iterations = tqdm(range(maxiter))
    # else:
    #     iterations = range(maxiter)
    iterations = range(maxiter)

    alphas = []
    lls = []
    window_height = 0
    Delta_new_ll = 0

    # ITERATIVE SEARCH
    ###########################################################################
    for i in iterations:
        print()
        print(f"Iterative search step {i+1}")

        # GENERATE ALPHA SAMPLE
        #######################################################################
        if i == 0:
            # 0: start with random search
            elem.set_alphaNP_init(0., sigalphainit)

            alphasamples = generate_alphaNP_sample(elem, nalphasamples_search,
                search_mode="random")

        # 1 -> break: grid search
        else:
            if i < maxiter:  # and Delta_new_ll < window_height / 20:
                alphasamples = generate_alphaNP_sample(
                    elem, nalphasamples_search, search_mode="grid")

            # else:
            #     # use old alphasamples
            #     print(f"BREAKING AT ITER: {i}, with maxiter: {maxiter}")
            #     break

        # ALPHAS FROM LAST ROUND -> OLD ALPHAS
        #######################################################################
        old_alphas = alphas
        old_lls = lls

        # COMPUTE LogL FOR NEWLY GENERATED ALPHA SAMPLES
        #######################################################################
        alphas, lls = compute_ll(elem, alphasamples, nelemsamples_search)

        minll_1 = np.percentile(lls, 1)

        if plot_output:
            delchisqlist = get_delchisq(lls, minll_1)
            plot_mc_output(alphas, delchisqlist,
                plotname=f"{i + 1}", minll=minll_1)

        # DEFINE POTENTIAL NEW INTERVAL
        #######################################################################
        (
            new_alphas, new_lls,
            new_best_alpha, new_lb_alpha, new_ub_alpha
        ) = get_new_alphaNP_interval(alphas, lls, scalefactor=scalefactor)

        # TEST NEW INTERVAL. IF GOOD, UPDATE ALPHA, SIGALPHA OF ELEM
        #######################################################################

        window_frac = 1 / 2
        window_lb = (1 - window_frac) / 2
        window_ub = (1 + window_frac) / 2

        window_width = max(new_alphas) - min(new_alphas)
        window_height = max(new_lls) - min(new_lls)

        (
            new_best_alpha, sig_new_best_alpha, Deltall
        ) = get_bestalpha_and_logL_spread_of_next_iteration(
            new_alphas, new_lls, alphas, lls
        )

        if (
                # Is new best alpha in centre of window?
                (
                    (min(new_alphas) + window_width * (1 - window_frac) / 2
                        < new_best_alpha)
                    and (new_best_alpha < (
                        min(new_alphas) + (1 + window_frac) / 2 * window_width))
                )

                and

                # Is the vertical spread small enough in the region of interest?
                (
                    Deltall < window_height / 10

                )
        ):

            new_alphas, new_lls = equilibrate_interval(new_alphas, new_lls,
                alpha_window_lfrac=window_lb, alpha_window_ufrac=window_ub)

            (
                Delta_new_ll, _, _, _
            ) = update_alphaNP_for_next_iteration(
                elem, new_alphas, new_lls, alphas, lls
            )

        else:
            (
                Delta_new_ll, _, _, _
            ) = update_alphaNP_for_next_iteration(
                elem, old_alphas, old_lls, alphas, lls
            )

            new_alphas = alphas
            new_lls = lls

            print(f"BREAKING AT ITER: {i}, with maxiter: {maxiter}")

            break

    # -1: final round: perform parabolic fits and use blocking method to

    allalphasamples = generate_alphaNP_sample(elem, nexps * nalphasamples_exp,
        search_mode="grid")

    # shuffle the sample
    np.random.shuffle(allalphasamples)

    delchisqs_exps = []

    for exp in range(nexps):
        # collect data for a single experiment
        alphasamples = allalphasamples[
            exp * nalphasamples_exp: (exp + 1) * nalphasamples_exp]

        # compute alphas and LLs for this experiment
        alphas, lls = compute_ll(elem, alphasamples, nelemsamples_exp)

        alphas_exps.append(alphas)
        bestalphas_exps.append(alphas[np.argmin(lls)])
        lls_exps.append(lls)

        minll_1 = np.percentile(lls, 1)
        delchisqlist = get_delchisq(lls, minll=minll_1)

        if plot_output:
            plot_mc_output(alphas, delchisqlist,
                plotname=f"m1_exp_{exp}", minll=minll_1)

        delchisqs_exps.append(delchisqlist)

    delchisqcrit = get_delchisq_crit(nsigmas=nsigmas, dof=1)

    confints_exps = np.array([get_confint(alphas_exps[s], delchisqs_exps[s],
        delchisqcrit) for s in range(nexps)])

    (best_alpha_pts, sig_alpha_pts,
        LB, sig_LB, UB, sig_UB) = \
        get_bestalphaNP_and_bounds(bestalphas_exps,
            confints_exps, nblocks=nblocks)

    elem.set_alphaNP_init(best_alpha_pts, sig_alpha_pts)

    if plot_output:
        plot_final_mc_output(elem, alphas_exps, delchisqs_exps,
            delchisqcrit,
            bestalphapt=best_alpha_pts, sigbestalphapt=sig_alpha_pts,
            lb=LB, siglb=sig_LB, ub=UB, sigub=sig_UB,
            nsigmas=nsigmas, xind=xind)

    return [
        np.array(alphas_exps), np.array(delchisqs_exps),
        delchisqcrit,
        best_alpha_pts, sig_alpha_pts,
        LB, sig_LB, UB, sig_UB,
        xind]
