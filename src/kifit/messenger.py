import os
import numpy as np
import json
from argparse import ArgumentParser


class RunParams:
    def __init__(self, runparams: ArgumentParser):
        self.__runparams = runparams

    @property
    def element_list(self):
        return self.__runparams.element_list

    @property
    def output_file_name(self):
        return self.__output_file_name

    @property
    def optimization_method(self):
        return self.__runparams.optimization_method

    @property
    def maxiter(self):
        return self.__runparams.maxiter

    @property
    def num_searches(self):
        return self.__runparams.num_searches

    @property
    def num_elemsamples_search(self):
        return self.__runparams.num_elemsamples_search

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
    def x0(self):
        return self.__runparams.x0

    @property
    def mphivar(self):
        return self.__runparams.mphivar

    @property
    def gkp_dims(self):
        return self.__runparams.gkp_dims

    @property
    def nmgkp_dims(self):
        return self.__runparams.nmgkp_dims

    @property
    def num_det_samples(self):
        return self.__runparams.num_det_samples

    @property
    def showalldetbounds(self):
        return self.__runparams.showalldetbounds

    @property
    def showbestdetbounds(self):
        return self.__runparams.showbestdetbounds

    @property
    def num_sigmas(self):
        return self.__runparams.num_sigmas

    @property
    def verbose(self):
        return self.__runparams.verbose


class Messenger(RunParams):

    def __init__(self, elemcollectionid, elemcollectionxrange, runparams):
        super().__init__(runparams)

        self.elem_collection_id = elemcollectionid
        self.init_x_range(elemcollectionxrange, runparams)

        self.runparams = runparams

        self.generate_result_folder()
        self.generate_output_data_folder()
        self.generate_plot_folder()

        self.fit_keys = []
        self.det_keys = []

    @classmethod
    def get(cls, elemcollectionid: str, elemcollectionxrange: list, runparams):
        return cls(elemcollectionid, elemcollectionxrange, runparams)

    def init_x_range(self, elemcollectionxrange, runparams):
        if runparams.mphivar is True:
            print("Kifit will run for all provided mphi values.")
            self.x_range = self.elem_collection_x_range
        else:
            print("Initialised x range: ", runparams.x0)
            self.x_range = runparams.x0

    def generate_result_folder(self):
        # path where to save results

        result_path = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))

        if not os.path.exists(result_path):
            print("Creating file at ", result_path)
            os.mkdir(result_path)

        self.result_path = result_path

        return result_path

    def generate_output_data_folder(self):
        # path where to save data

        outputdatapath = os.path.join(self.result_path, "output_data")

        if not os.path.exists(outputdatapath):
            print("Creating file at ", outputdatapath)
            os.mkdir(outputdatapath)

        self.output_data_path = outputdatapath

        return outputdatapath

    def generate_plot_folder(self):
        # path where to save plots

        plot_path = os.path.join(self.result_path, "plots")

        if not os.path.exists(plot_path):
            print("Creating file at ", plot_path)
            os.mkdir(plot_path)

        self.plot_path = plot_path

        return plot_path

    def generate_plot_path(self, plotname, elemid=None, xind=None):
        return os.path.join(self.plot_path,
            f"{plotname}_"
            + (f"{self.elem_collection_id}" if elemid is None else f"{elemid}")
            + (f"_x{str(xind)}" if xind is not None else "")
            + ".png")

    def fit_output_path(self, xind):
        # path where to save fit results for x=xind
        fit_output_path = os.path.join(
            self.output_data_path, (
                f"{self.elem_collection_id}_"
                + f"{self.runparams.optimization_method}_"
                + f"{self.runparams.num_searches}searches_"
                + f"{self.runparams.num_elemsamples_search}es-search_"
                + f"{self.runparams.num_exp}exps_"
                + f"{self.runparams.num_elemsamples_exp}es-exp_"
                + f"{self.runparams.num_alphasamples_exp}as-exp_"
                + f"maxiter{self.runparams.maxiter}_"
                + f"blocksize{self.runparams.block_size}_"
                + f"x{xind}.json")
        )

        return fit_output_path

    def det_output_path(self, gkp, dim, xind):
        """
        Path where to save det results for the element defined by elemstr.
        dim = dimension of det
        gkp -> generalised King plot if True, else no-mass generalised King plot
        x = x-coefficient index

        """
        det_output_path = os.path.join(
            self.output_data_path, (
                f"{self.elem_collection_id}_"
                + f"{dim}-dim_"
                + ("gkp_" if gkp else "nmgkp_")
                + f"{self.runparams.num_det_samples}samples_"
                + f"x{xind}.json")
        )

        return det_output_path

    def write_to_path(self, path, keys, results):

        assert len(results) == len(keys), (len(results), len(keys))

        res_dict = {key: [] for key in keys}

        for i, res in enumerate(results):
            if isinstance(res, np.ndarray):
                reslist = res.tolist()
                res_dict[keys[i]].append(reslist)
            else:
                res_dict[keys[i]] = res

        with open(path, 'w') as json_file:
            json.dump(res_dict, json_file)

        return 0

    def read_from_path(self, path, keys):
        if os.path.exists(path):
            print("Loading data from ", path)

            with open(path) as json_file:
                res = json.load(json_file)

                # res_list = []
                # for key in keys:
                #     res_list.append(res[key])
                for key in keys:
                    if isinstance(res[key], list):
                        res[key] = np.array(res[key])

        return res  # _list

    def set_fit_keys(self, keys):

        self.fit_keys = keys

    def set_det_keys(self, keys):
        self.det_keys = keys

    def read_fit_output(self, x):
        return self.read_from_path(self.fit_output_path(x), self.fit_keys)

    def read_det_output(self, gkp, dim, x):
        return self.read_from_path(self.det_output_path(gkp, dim, x), self.det_keys)

    def write_fit_output(self, x, results):
        self.write_to_path(self.fit_output_path(x), self.fit_keys, results)

    def write_det_output(self, gkp, dim, x, results):
        self.write_to_path(self.det_output_path(gkp, dim, x), self.det_keys, results)
