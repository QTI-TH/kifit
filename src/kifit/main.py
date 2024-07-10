import argparse

import datetime
from kifit.loadelems import Elem
from kifit.performfit_new import sample_alphaNP_fit, generate_path
from kifit.plotfit import plot_linfit, plot_alphaNP_ll

# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')


def main(args):
    """Determine alphaNP bounds given elements data."""
    # define output folder's name

    element_collection = []
    for elem in args.elements_list:
        element_collection.append(Elem.get(str(elem)))

    output_filename = (
        f"{args.outputfile_name}_{args.optimization_method}"
        f"_{args.num_searches}searches_{args.num_experiments}nexps"
        f"_{args.num_elemsamples_search}es_{args.num_elemsamples_exp}ee"
        f"_{args.num_alphasamples_exp}ae"
    )
    
    _, plot_path = generate_path(output_filename)

    mc_output = sample_alphaNP_fit(
        element_collection,
        output_filename=output_filename,
        nsearches=args.num_searches,
        nelemsamples_search=args.num_elemsamples_search,
        nexps=args.num_experiments,
        nelemsamples_exp=args.num_elemsamples_exp,
        nalphasamples_exp=args.num_alphasamples_exp,
        nblocks=args.num_blocks,
        maxiter=args.maxiter,
        mphivar=False,
        plot_output=True,
        opt_method=args.optimization_method,
        x0=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boosting VQE with DBI.")
    parser.add_argument(
        "--elements_list", 
        type=list_of_strings, 
        help="List of strings corresponding to names of data folders",
    )
    parser.add_argument(
        "--outputfile_name", 
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
        "--num_experiments", 
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
        "--num_blocks",
        default=100, 
        type=int, 
        help="# blocks used to perform the blocking method, which returns the bounds estimation",
    )
    parser.add_argument(
        "--num_samples_det",
        default=100, 
        type=int, 
        help="# generated samples executing determinant method",
    )

    args = parser.parse_args()
    main(args)
