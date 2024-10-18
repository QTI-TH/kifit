import os
import json
import logging
import numpy as np

from kifit.build import ElemCollection
from kifit.fitools import fit_keys, sample_alphaNP_fit
from kifit.detools import det_keys, sample_alphaNP_det
from kifit.plot import plot_linfit, plot_alphaNP_ll, plot_mphi_alphaNP
from kifit.config import RunParams, Paths, Config


logging.basicConfig(level=logging.INFO)
np.random.seed(1)


class Runner:

    def __init__(self,
            config: Config,
            collection: ElemCollection):

        self.config = config
        self.collection = collection

    @classmethod
    def build(cls):
        params = RunParams()

        collection = ElemCollection(
            params.element_list,
            params.gkp_dims,
            params.nmgkp_dims
        )

        paths = Paths(params, collection.id, fit_keys, det_keys)

        print("run.py collection.x_vals", collection.x_vals)

        config = Config(params, paths, collection.x_vals)

        return cls(config, collection)

    def generate_all_King_plots(self):

        for elem in self.collection.elems:
            plot_linfit(elem, self.config)

    def run(self):

        self.generate_all_King_plots()

        # elem = None
        #
        # if self.collection.len == 1:
        #
        #     elem = self.collection.elems[0]

        for elem in self.collection.elems:

            for x in self.config.x_vals_det:
                elem._update_Xcoeffs(x)
                for dim in self.config.params.gkp_dims:

                    sample_alphaNP_det(
                        elem=elem,
                        messenger=self.config,
                        dim=dim,
                        detstr="gkp",
                        xind=x)

                for dim in self.config.params.nmgkp_dims:

                    sample_alphaNP_det(
                        elem=elem,
                        messenger=self.config,
                        dim=dim,
                        detstr="nmgkp",
                        xind=x)

                for dim in self.config.params.proj_dims:

                    sample_alphaNP_det(
                        elem=elem,
                        messenger=self.config,
                        dim=dim,
                        detstr="proj",
                        xind=x)

        for x in self.config.x_vals_fit:

            for elem in self.collection.elems:
                elem._update_Xcoeffs(x)

            sample_alphaNP_fit(
                self.collection,
                self.config,
                xind=x
            )

            plot_alphaNP_ll(
                self.collection,
                messenger=self.config,
                expstr="search",
                logplot=True,
                xind=x)

            plot_alphaNP_ll(
                self.collection,
                messenger=self.config,
                expstr="experiment",
                xind=x)

        if len(self.config.x_vals_fit) > 1 or len(self.config.x_vals_det) > 1:

            plot_mphi_alphaNP(
                elem_collection=self.collection,
                messenger=self.config)

    def generate_all_alphaNP_ll_plots(self):

        for x in self.config.x_vals_fit:

            plot_alphaNP_ll(
                self.collection,
                messenger=self.config,
                expstr="search",
                logplot=True,
                xind=x)

            plot_alphaNP_ll(
                self.collection,
                messenger=self.config,
                expstr="experiment",
                xind=x)

    def generate_mphi_alphaNP_plot(self):

        if len(self.config.x_vals_fit) > 1 or len(self.config.x_vals_det) > 1:
            plot_mphi_alphaNP(
                elem_collection=self.collection,
                messenger=self.config)

    def print_relative_uncertainties(self):

        for elem in self.collection.elems:
            print("Element: ", elem.id)
            elem.print_relative_uncertainties

    def dump_config(self, filepath, overwrite=False):
        """Dump configuration of the Runner into a json file located in `filepath`."""

        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(f"File '{filepath}' already exists. Use overwrite=True to overwrite it.")

        runparams_dict = vars(self.config.params._RunParams__runparams)
        formatted_dict = {key: value for key, value in runparams_dict.items()}

        with open(filepath, 'w') as json_file:
            json.dump(formatted_dict, json_file, indent=4)

        return formatted_dict
    

    def load_config(self, filepath):
        """Load configuration of a Runner from `filepath` and set it as config."""
        
        with open(filepath, 'r') as json_file:
            config_data = json.load(json_file)

