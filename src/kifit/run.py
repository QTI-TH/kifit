import os
import json
import logging
import numpy as np

from kifit.build import ElemCollection
from kifit.fitools import fit_keys, sample_alphaNP_fit
from kifit.detools import det_keys, sample_alphaNP_det
from kifit.plot import (
    plot_linfit,
    plot_search_window,
    plot_alphaNP_ll,
    plot_alphaNP_ll_zoom,
    plot_mphi_alphaNP,
)
from kifit.config import RunParams, Paths, Config


logging.basicConfig(level=logging.INFO)
np.random.seed(1)


class Runner:

    def __init__(self, run_params: RunParams):

        # construct a collection of elements
        self.collection = ElemCollection(
            run_params.element_list, run_params.reference_transitions
        )

        print("elem collection elem list")
        print(run_params.element_list)

        # set the config attribute
        paths = Paths(run_params, self.collection.id, fit_keys, det_keys)
        self.config = Config(run_params, paths, self.collection.x_vals)

    def generate_all_King_plots(self):
        for elem in self.collection.elems:
            plot_linfit(elem, self.config)

    def run(self):

        self.generate_all_King_plots()

        for elem in self.collection.elems:

            elem.check_det_dims(
                self.config.params.gkp_dims,
                self.config.params.nmgkp_dims,
                self.config.params.proj_dims,
            )

        for elem in self.collection.elems:

            for x in self.config.x_vals_det:
                elem._update_Xcoeffs(x)
                for dim in self.config.params.gkp_dims:

                    sample_alphaNP_det(
                        elem=elem, messenger=self.config, dim=dim, detstr="gkp", xind=x
                    )

                for dim in self.config.params.nmgkp_dims:

                    sample_alphaNP_det(
                        elem=elem,
                        messenger=self.config,
                        dim=dim,
                        detstr="nmgkp",
                        xind=x,
                    )

                for dim in self.config.params.proj_dims:

                    sample_alphaNP_det(
                        elem=elem, messenger=self.config, dim=dim, detstr="proj", xind=x
                    )

        for x in self.config.x_vals_fit:

            for elem in self.collection.elems:
                elem._update_Xcoeffs(x)

            sample_alphaNP_fit(self.collection, self.config, xind=x)

            # plot_alphaNP_ll(
            #     self.collection,
            #     messenger=self.config,
            #     expstr="search",
            #     logplot=True,
            #     xind=x)

            plot_alphaNP_ll(
                self.collection,
                messenger=self.config,
                # expstr="experiment",
                xind=x,
            )

            if x == 0:
                plot_alphaNP_ll_zoom(self.collection, messenger=self.config, xind=x)

        if len(self.config.x_vals_fit) > 1 or len(self.config.x_vals_det) > 1:

            plot_mphi_alphaNP(elem_collection=self.collection, messenger=self.config)

    def generate_all_alphaNP_ll_plots(self):

        for x in self.config.x_vals_fit:

            plot_search_window(messenger=self.config, xind=x)

            plot_alphaNP_ll(
                self.collection,
                messenger=self.config,
                # expstr="experiment",
                xind=x,
            )

            if x == 0:

                plot_alphaNP_ll_zoom(self.collection, messenger=self.config, xind=x)

    def generate_mphi_alphaNP_plot(self):

        if len(self.config.x_vals_fit) > 1 or len(self.config.x_vals_det) > 1:
            plot_mphi_alphaNP(elem_collection=self.collection, messenger=self.config)

    def print_relative_uncertainties(self):

        for elem in self.collection.elems:
            print("Element: ", elem.id)
            elem.print_relative_uncertainties

    def dump_config(self, filepath, overwrite=False):
        """Dump configuration of the Runner into a json file located in `filepath`."""

        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(
                f"File '{filepath}' already exists. Use overwrite=True to overwrite it."
            )

        runparams_dict = vars(self.config.params._RunParams__runparams)
        formatted_dict = {key: value for key, value in runparams_dict.items()}

        with open(filepath, "w") as json_file:
            json.dump(formatted_dict, json_file, indent=4)

        return formatted_dict

    def load_config(self, configuration_file: str):
        """Load configuration of a Runner from `filepath` and set it as config."""

        run_params = RunParams(configuration_file=configuration_file)

        # construct a collection of elements
        self.collection = ElemCollection(
            run_params.element_list  # ,
            # run_params.gkp_dims,
            # run_params.nmgkp_dims
        )

        # set the config attribute
        paths = Paths(run_params, self.collection.id, fit_keys, det_keys)
        self.config = Config(run_params, paths, self.collection.x_vals)
