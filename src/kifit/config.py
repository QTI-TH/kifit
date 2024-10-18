import os
import json
import numpy as np
import logging
from argparse import ArgumentParser, ArgumentTypeError


def check_input_logrid_frac(value):
    intval = int(value)

    if intval > 0:
        raise ArgumentTypeError(
            f"""{value} is not a valid value for logrid_frac.
            Provide negative integer.""")
    elif intval < -10:
        raise ArgumentTypeError(
            f"""{value} seems like a ridiculously small number for logrid_frac.
            Give it a larger (but still negative) integer value.""")
    return intval


class RunParams:
    def __init__(self):
        self.__runparams = self.parse_arguments()

    @property
    def element_list(self):
        return self.__runparams.element_list

    @property
    def num_alphasamples_search(self):
        return self.__runparams.num_alphasamples_search

    @property
    def num_elemsamples_per_alphasample_search(self):
        return self.__runparams.num_elemsamples_per_alphasample_search

    @property
    def search_mode(self):
        return self.__runparams.search_mode

    # @property
    # def num_optigrid_searches(self):
    #     return self.__runparams.num_optigrid_searches

    @property
    def logrid_frac(self):
        return self.__runparams.logrid_frac

    @property
    def num_exp(self):
        return self.__runparams.num_exp

    @property
    def num_elemsamples_exp(self):
        return self.__runparams.num_elemsamples_exp

    @property
    def num_alphasamples_exp(self):
        return self.__runparams.num_alphasamples_exp

    @property
    def block_size(self):
        return self.__runparams.block_size

    @property
    def min_percentile(self):
        return self.__runparams.min_percentile

    @property
    def x0_fit(self):
        return self.__runparams.x0_fit

    @property
    def x0_det(self):
        return self.__runparams.x0_det

    @property
    def mphivar_fit(self):
        return self.__runparams.mphivar_fit

    @property
    def mphivar_det(self):
        return self.__runparams.mphivar_det

    @property
    def gkp_dims(self):
        return self.__runparams.gkp_dims

    @property
    def nmgkp_dims(self):
        return self.__runparams.nmgkp_dims

    @property
    def proj_dims(self):
        return self.__runparams.proj_dims

    @property
    def num_det_samples(self):
        return self.__runparams.num_det_samples

    @property
    def showalldetbounds(self):
        return self.__runparams.showalldetbounds

    @property
    def showalldetvals(self):
        return self.__runparams.showalldetvals

    @property
    def showbestdetbounds(self):
        return self.__runparams.showbestdetbounds

    @property
    def num_sigmas(self):
        return self.__runparams.num_sigmas

    @property
    def verbose(self):
        return self.__runparams.verbose

    @staticmethod
    def parse_arguments():
        parser = ArgumentParser(
            description="Computing King plot bounds on new atomic Yukawa potential.")

        parser.add_argument(
            "--element_list",
            nargs="+",
            type=str,
            help="List of strings corresponding to names of data folders",
        )
        parser.add_argument(
            "--num_alphasamples_search",
            default=1000,
            type=int,
            help="no. alphaNP samples generated during each search step",
        )
        parser.add_argument(
            "--num_elemsamples_per_alphasample_search",
            default=100,
            type=int,
            help="""no. element samples generated for each alphaNP sample during
            initial search""",
        )
        parser.add_argument(
            "--search_mode",
            default="detlogrid",
            choices=["detlogrid", "globalogrid"],  # , "optigrid"],
            help="""method used during search phase. logrid uses input from determinant methods, globalopt does not""",
        )
        parser.add_argument(
            "--logrid_frac",
            default=-5,
            type=check_input_logrid_frac,
            help="""log10 of fraction defining the alphaNP scan region for
            initial search. Should be a negative or zero integer: 0 -> scan
            region -> 0, -infty -> scan region infinite. Please provide integers
            larger or equal to -10.""",
        )
        parser.add_argument(
            "--num_exp",
            default=10,
            type=int,
            help="no. experiments after the optimal working window has been found",
        )
        parser.add_argument(
            "--num_elemsamples_exp",
            default=100,
            type=int,
            help="no. element samples generated during each experiment",
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
            choices=range(1, 50),
            type=int,
            help="Min percentile value used to compute a robust estimation of min(logL)",
        )
        parser.add_argument(
            "--x0_fit",
            nargs="+",
            default=[0],
            type=int,
            help="Target mphi indices for fit",
        )
        parser.add_argument(
            "--x0_det",
            nargs="+",
            default=[0],
            type=int,
            help="Target mphi indices for determinants",
        )
        parser.add_argument(
            "--mphivar_fit",
            action="store_true",
            help="If specified, the fit is performed for all mphi values.",
        )
        parser.add_argument(
            "--mphivar_det",
            action="store_true",
            help="If specified, the determinants are computed for all mphi values.",
        )
        parser.add_argument(
            "--gkp_dims",
            nargs="+",
            default=[3],
            type=int,
            help="List of generalised King plot dimensions",
        )
        parser.add_argument(
            "--nmgkp_dims",
            nargs="+",
            default=[3],
            type=int,
            help="List of no-mass generalised King plot dimensions",
        )
        parser.add_argument(
            "--proj_dims",
            nargs="+",
            default=[3],
            type=int,
            help="List of projection method dimensions",
        )
        parser.add_argument(
            "--num_det_samples",
            default=100,
            type=int,
            help="Number of alpha/element samples generated for the det bounds"
        )
        parser.add_argument(
            "--showalldetbounds",
            action="store_true",
            help="If true, the det bounds are shown for all combinations of the data."
        )
        parser.add_argument(
            "--showalldetvals",
            action="store_true",
            help="If true, the det values +/- uncertainties (1 sigma) are plotted."
        )
        parser.add_argument(
            "--showbestdetbounds",
            action="store_true",
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
            help="""If specified, extra statements are printed and extra plots
            plotted."""
        )

        return parser.parse_args()
    
    def load_arguments_from_file(self, json_file):
        """Load arguments from a JSON file and set them as attributes."""
        # Get the default arguments from the parser first
        defaults = vars(self.parse_arguments())
        
        # Load the JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Override defaults with any values in the JSON file
        for key, value in data.items():
            if key in defaults:
                defaults[key] = value
        
        # Set the attributes in the __runparams
        self.__runparams = type('Args', (), defaults)


class Paths:
    def __init__(self, params: RunParams, collectionid: str,
            fit_keys: list, det_keys: list):
        self.__params = params
        self.__elem_collection_id = collectionid
        self.generate_result_folder()
        self.generate_output_data_folder()
        self.generate_plot_folder()
        self.fit_keys = fit_keys
        self.det_keys = det_keys

    def generate_result_folder(self):
        # path where to save results

        result_path = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))

        if not os.path.exists(result_path):
            logging.info("Creating file at ", result_path)
            os.mkdir(result_path)

        self.result_path = result_path

        return result_path

    def generate_output_data_folder(self):
        # path where to save data

        outputdatapath = os.path.join(self.result_path, "output_data")

        if not os.path.exists(outputdatapath):
            logging.info("Creating file at ", outputdatapath)
            os.mkdir(outputdatapath)

        self.output_data_path = outputdatapath

        return outputdatapath

    def generate_plot_folder(self):
        # path where to save plots

        plot_path = os.path.join(self.result_path, "plots")

        if not os.path.exists(plot_path):
            logging.info("Creating file at ", plot_path)
            os.mkdir(plot_path)

        self.plot_path = plot_path

        return plot_path

    def generate_plot_path(self, plotname, elemid=None, xind=None):
        return os.path.join(self.plot_path,
            f"{plotname}_"
            + (f"{self.__elem_collection_id}" if elemid is None else f"{elemid}")
            + (f"_x{str(xind)}" if xind is not None else "")
            + ".png")

    def search_output_path(self, xind):
        # path where to save fit results for x=xind
        search_output_path = os.path.join(
            self.output_data_path, (
                "search_output_"
                + f"{self.__elem_collection_id}_"
                + f"{self.__params.num_elemsamples_per_alphasample_search}es-search_"
                + f"{self.__params.num_alphasamples_search}as-search_"
                + f"{self.__params.search_mode}-search_"
                + (f"{self.__params.logrid_frac}logridfrac_"
                    if self.__params.search_mode == "detlogrid" else "")
                # + (f"{self.__params.num_optigrid_searches}searches_"
                #     if self.__params.search_mode == "optigrid" else "")
                + f"{self.__params.num_exp}exps_"
                + f"{self.__params.num_elemsamples_exp}es-exp_"
                + f"{self.__params.num_alphasamples_exp}as-exp_"
                + f"{self.__params.min_percentile}minperc_"
                + f"blocksize{self.__params.block_size}_"
                + f"x{xind}.json")
        )

        return search_output_path

    def fit_output_path(self, xind):
        # path where to save fit results for x=xind
        fit_output_path = os.path.join(
            self.output_data_path, (
                f"{self.__elem_collection_id}_"
                + f"{self.__params.num_elemsamples_per_alphasample_search}es-search_"
                + f"{self.__params.num_alphasamples_search}as-search_"
                + f"{self.__params.search_mode}-search_"
                + (f"{self.__params.logrid_frac}logridfrac_"
                    if self.__params.search_mode == "detlogrid" else "")
                # + (f"{self.__params.num_optigrid_searches}searches_"
                #     if self.__params.search_mode == "optigrid" else "")
                + f"{self.__params.num_exp}exps_"
                + f"{self.__params.num_elemsamples_exp}es-exp_"
                + f"{self.__params.num_alphasamples_exp}as-exp_"
                + f"{self.__params.min_percentile}minperc_"
                + f"blocksize{self.__params.block_size}_"
                + f"x{xind}.json"
            )
        )

        return fit_output_path

    def det_output_path(self, detstr, dim, xind):
        """
        Path where to save det results for the element defined by elemstr.
        dim = dimension of det
        gkp -> generalised King plot if True, else no-mass generalised King plot
        x = x-coefficient index

        """
        det_output_path = os.path.join(
            self.output_data_path, (
                f"{self.__elem_collection_id}_"
                + f"{dim}-dim_" + str(detstr) + "_"
                + f"{self.__params.num_det_samples}samples_"
                + f"x{xind}.json")
        )

        return det_output_path

    def write_to_path(self, path, keys, results):

        assert len(results) == len(keys), (len(results), len(keys))

        res_dict = {}  # {key: [] for key in keys}

        for i, res in enumerate(results):
            if isinstance(res, np.ndarray):
                reslist = res.tolist()
                res_dict[keys[i]] = reslist
                # res_dict[keys[i]].append(reslist)
            else:
                res_dict[keys[i]] = res

        with open(path, 'w') as json_file:
            json.dump(res_dict, json_file)

        return 0

    def read_from_path(self, path, keys):
        if os.path.exists(path):
            logging.info("Loading data from %s", path)

            with open(path) as json_file:
                res = json.load(json_file)

                for key in keys:
                    if isinstance(res[key], list):
                        res[key] = np.array(res[key])
                return res
        else:
            raise ImportError(f"{path} Does not yet exist. Please run Kifit.")

    def read_search_output(self, x):
        return self.read_from_path(self.search_output_path(x), self.fit_keys)

    def read_fit_output(self, x):
        return self.read_from_path(self.fit_output_path(x), self.fit_keys)

    def read_det_output(self, detstr, dim, x):
        return self.read_from_path(self.det_output_path(detstr, dim, x), self.det_keys)

    def write_search_output(self, x, results):
        self.write_to_path(self.search_output_path(x), self.fit_keys, results)

    def write_fit_output(self, x, results):
        self.write_to_path(self.fit_output_path(x), self.fit_keys, results)

    def write_det_output(self, detstr, dim, x, results):
        self.write_to_path(self.det_output_path(detstr, dim, x), self.det_keys, results)


class Config:

    def __init__(self,
            runparams: RunParams,
            paths: Paths,
            collectionxvals):

        self.params = runparams
        assert isinstance(runparams, RunParams)
        self.paths = paths
        assert isinstance(paths, Paths)
        print("paths", paths.fit_output_path(0))

        self.__init_x_vals(collectionxvals)

    def __init_x_vals(self, collectionxvals):
        if self.params.mphivar_fit is True:
            logging.info("Kifit will run fit for all provided mphi values.")
            self.x_vals_fit = collectionxvals
        else:
            if not set(self.params.x0_fit) <= set(collectionxvals):
                print("collectionxvals", collectionxvals)
                print("x0_fit         ", self.params.x0_fit)
                raise IndexError(r"Parsed invalid x0_fit index.")
            logging.info("Initialised x range for fit: %s", self.params.x0_fit)
            self.x_vals_fit = self.params.x0_fit

        if self.params.mphivar_det is True:
            logging.info("Kifit will run dets for all provided mphi values.")
            self.x_vals_det = collectionxvals
        else:
            if not set(self.params.x0_det) <= set(collectionxvals):
                raise IndexError("Parsed invalid x0_det index.")
            logging.info("Initialised x range for determinants: %s", self.params.x0_det)
            self.x_vals_det = self.params.x0_det

    
