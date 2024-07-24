import argparse

from kifit.builder import ElemCollection
from kifit.messenger import Messenger
from kifit.hunter import sample_alphaNP_fit, sample_alphaNP_det
from kifit.artist import plot_linfit, plot_alphaNP_ll  # , plot_mphi_alphaNP


def main(args):
    """Determine alphaNP bounds given elements data."""

    collection = ElemCollection.get(args.element_list)
    messenger = Messenger.get(collection.id, collection.x_range, args)

    collection.check_det_dims(args.gkp_dims, args.nmgkp_dims)

    for elem in collection.elems:
        plot_linfit(elem, messenger)

    for x in messenger.x_range:

        for elem in collection.elems:
            elem._update_Xcoeffs(x)

        _ = sample_alphaNP_fit(
            collection,
            messenger,
            xind=x
        )

        if collection.len == 1:

            elem = collection.elems[0]

            for dim in args.gkp_dims:

                _ = sample_alphaNP_det(
                    elem=elem,
                    messenger=messenger,
                    dim=dim,
                    gkp=True,
                    xind=x)

            for dim in args.nmgkp_dims:

                _ = sample_alphaNP_det(
                    elem=elem,
                    messenger=messenger,
                    dim=dim,
                    gkp=False,
                    xind=x)

        plot_alphaNP_ll(
            collection,
            messenger=messenger,
            xind=x)

###############################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boosting VQE with DBI.")
    parser.add_argument(
        "--element_list",
        nargs="+",
        type=str,
        help="List of strings corresponding to names of data folders",
    )
    parser.add_argument(
        "--output_file_name",
        default="PippoCamino",
        type=str,
        help="Name of the output folder",
    )
    parser.add_argument(
        "--optimization_method",
        default="Powell",
        type=str,
        help="Optimization method used to find the best experiment window",
    )
    parser.add_argument(
        "--maxiter",
        default=1000,
        type=int,
        help="Max number of iterations for optimization early stopping",
    )
    parser.add_argument(
        "--num_searches",
        default=10,
        type=int,
        help="# searches (optimizations) run to find the optimal working window",
    )
    parser.add_argument(
        "--num_elemsamples_search",
        default=100,
        type=int,
        help="# generated elements during each search step",
    )
    parser.add_argument(
        "--num_exp",
        default=10,
        type=int,
        help="# experiments after the optimal working window has been found",
    )
    parser.add_argument(
        "--num_elemsamples_exp",
        default=100,
        type=int,
        help="# generated elements during each final experiment",
    )
    parser.add_argument(
        "--num_alphasamples_exp",
        default=100,
        type=int,
        help="# generated alpha NP during each final experiment",
    )
    parser.add_argument(
        "--block_size",
        default=10,
        type=int,
        help="Size of the blocks used to perform the blocking method",
    )
    parser.add_argument(
        "--min_percentile",
        default=1,
        choices=range(1, 100),
        type=float,
        help="Min percentile value used to compute a robust estimation of min(logL)",
    )
    parser.add_argument(
        "--x0",
        nargs="+",
        default=[0],
        type=int,
        help="Target mphi indices",
    )
    parser.add_argument(
        "--mphivar",
        action="store_true",
        help="If specified, a loop is performed over all the mphi values in the datafile",
    )
    parser.add_argument(
        "--gkp_dims",
        nargs="+",
        default=[],
        type=int,
        help="List of generalised King plot dimensions",
    )
    parser.add_argument(
        "--nmgkp_dims",
        nargs="+",
        default=[],
        type=int,
        help="List of no-mass generalised King plot dimensions",
    )
    parser.add_argument(
        "--num_det_samples",
        default=100,
        type=int,
        help="Number of alpha/element samples generated for the det bounds"
    )
    parser.add_argument(
        "--showalldetbounds",
        default="false",
        type=str,
        help="If true, the det bounds are shown for all combinations of the data."
    )
    parser.add_argument(
        "--showbestdetbounds",
        default="true",
        type=str,
        help="If true, the best det bounds are shown."
    )
    parser.add_argument(
        "--num_sigmas",
        default=2,
        type=int,
        help="Number of sigmas for which the bounds are calculated."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If specified, extra statements are printed."
    )

    args = parser.parse_args()
    main(args)
