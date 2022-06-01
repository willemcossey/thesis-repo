from helper.Distribution import Uniform
from helper.Datapoint import Datapoint
import numpy as np
from math import sqrt
import json
import hashlib

# from scipy.stats.qmc import Sobol, Halton
from threading import Thread
from tqdm import tqdm
from os.path import exists
import random

# Create a dataset object. A dataset is a collection of Datapoint objects. It is a collection of input-output pairs. The object only points to the location where its datapoints are stored.
class Dataset:
    def __init__(
        self,
        rng,
        bounds,
        assumptions,
        children=None,
        size=1,
        datapoints=[],
        name=None,
        lazy=False,
    ):
        self.meta = {
            "rng": rng,
            "domain_bounds": bounds,
            "size": size,
            "children": [],
            "experiment_assumptions": assumptions,
        }
        print(self.meta["rng"])
        self.datapoints = datapoints
        if children is not None and not lazy:
            self.import_datasets(children)
        if len(datapoints) == 0:
            self.generate_input()
        assert len(self.datapoints) == self.meta["size"]
        if name is None:
            self.name = hashlib.md5(
                str(sorted(self.meta.items())).encode("utf-8")
            ).hexdigest()
        else:
            self.name = name
        self.save()
        if not (lazy or exists(f"""src\\datasets\\cache\\{self.name}.npz""")):
            self.save(ftype="npz")

    # Return the number of datapoints contained in the dataset
    def get_size(self):
        return len(self.datapoints)

    # Generate a number of datapoints that only have inputs. The inputs are computed according to the 'rng' argument specified.
    def generate_input(self):
        inp = np.array([])
        if self.meta["rng"] == "random":
            inp = np.array(
                [
                    Uniform(
                        self.meta["domain_bounds"][i][0],
                        self.meta["domain_bounds"][i][1],
                    ).sample(self.meta["size"])
                    for i in self.meta["domain_bounds"].keys()
                ]
            )
        # elif self.meta["rng"] == "sobol":
        #     gen = Sobol(len(self.meta["domain_bounds"].keys()))
        #     points = gen.random(self.size)
        #     inp = np.transpose(
        #         np.array(
        #             [
        #                 points[:, i]
        #                 * (
        #                     self.meta["domain_bounds"][i][1]
        #                     - self.meta["domain_bounds"][i][0]
        #                 )
        #                 + self.meta["domain_bounds"][i][0]
        #                 for i in self.meta["domain_bounds"].keys()
        #             ]
        #         )
        #     )
        # elif self.meta["rng"] == "halton":
        #     gen = Halton(len(self.meta["domain_bounds"].keys()))
        #     points = gen.random(self.size)
        #     inp = np.transpose(
        #         np.array(
        #             [
        #                 points[:, i]
        #                 * (
        #                     self.meta["domain_bounds"][i][1]
        #                     - self.meta["domain_bounds"][i][0]
        #                 )
        #                 + self.meta["domain_bounds"][i][0]
        #                 for i in self.meta["domain_bounds"].keys()
        #             ]
        #         )
        #     )
        elif self.meta["rng"] == "single":
            inp = np.array(
                [
                    Uniform(
                        self.meta["domain_bounds"][i][0],
                        self.meta["domain_bounds"][i][1],
                    ).sample(1)
                    for i in self.meta["domain_bounds"].keys()
                ]
                * self.meta["size"]
            ).reshape(2, -1, order="F")
        else:
            raise ValueError
        for n in range(self.meta["size"]):
            lmb = float(inp[0, n])
            m = float(inp[1, n])
            theta_std = sqrt(self.meta["experiment_assumptions"]["gamma"] * lmb)
            ass = {
                **self.meta["experiment_assumptions"],
                **{"lmb": lmb, "mean_opinion": m, "theta_std": theta_std},
            }
            if self.meta["rng"] == "single":
                ass["seed"] = random.randint(1, 2 ** 32 - 1)
            dp = Datapoint(
                dict(zip(self.meta["domain_bounds"].keys(), inp[:, n].tolist())), ass
            )
            self.datapoints.append(f"""{dp.name}.json""")
        self.meta["size"] = self.get_size()
        pass

    # Import the datapoints from the datasets specified in 'child_name_array' into the current dataset object. Used to merge existing datasets.
    def import_datasets(self, child_name_array):
        # load datasets
        child_array = []
        for name in child_name_array:
            child_array.append(Dataset.from_json(f"""./src/datasets/{name}"""))
        # check if datasets where generated by same rng
        for d in child_array:
            assert d.meta["rng"] == self.meta["rng"]
            # check if datasets have same bounds to conserve discrepancy
            assert d.meta["domain_bounds"] == self.meta["domain_bounds"]
            assert [
                d.meta["experiment_assumptions"][i]
                == self.meta["experiment_assumptions"][i]
                or self.meta["experiment_assumptions"][i] != "seed"
                for i in self.meta["experiment_assumptions"]
            ]
            # add members of datasets to members
            for dp_name in d.datapoints:
                if dp_name not in self.datapoints:
                    self.datapoints.append(dp_name)
            # add dataset names to children
            self.meta["children"].append(f"{d.name}.json")
        # update size
        self.meta["size"] = self.get_size()

    # Compute the outputs of the datapoints
    def compute_output(self):
        for dp_name in self.datapoints:
            dp = Datapoint.from_json(f"""./src/datapoints/{dp_name}""")
            if (dp.output is None) or (dp.output["raw"] is None):
                dp.compute_output()
        self.save(ftype="npz")

    # Save the dataset. A thread is used to prevent file corruption in case of an error.
    def save(self, ftype="json"):
        t = Thread(target=self._save(ftype=ftype))
        t.start()
        t.join()
        pass

    # Save the object as a json file.
    def _save(self, ftype):
        if ftype == "json":
            with open(f"""./src/datasets/{self.name}.json""", mode="w") as file:
                json.dump(self.to_json(), file, indent=1)
                file.flush()
                file.close()
                pass
        elif ftype == "npz":
            inputs = self.get_inputs(silent=False)
            outputs = self.get_outputs(silent=False, lazy=False)
            np.savez(
                f"""./src/datasets/cache/{self.name}.npz""",
                inputs=inputs,
                outputs=outputs,
            )
        else:
            raise ValueError

    # Compute an aggregated version of the outputs. In this case a histogram is constructed. The histograms are normalized and sums to 1.
    def compute_aggregated_output(self, n):
        # Compute the effective 'histogram' of the solution over n equispaced intervals.
        for dp_name in tqdm(self.datapoints):
            try:
                dp = Datapoint.from_json(f"""./src/datapoints/{dp_name}""")
                if dp.output is None:
                    dp.compute_output()
                dp.compute_aggregated_output(n)
            except (json.decoder.JSONDecodeError):
                print(f"Something's wrong with this json file: {dp_name}")
        self.save(ftype="npz")

    def to_json(self):
        return self.__dict__

    # Load the object from a file. If 'lazy' is True, The datapoint outputs are not recomputed.
    @staticmethod
    def from_json(filename, lazy=False):
        f = open(filename, "r+")
        set = json.load(f)
        f.flush()
        f.close()
        return Dataset(
            set["meta"]["rng"],
            set["meta"]["domain_bounds"],
            set["meta"]["experiment_assumptions"],
            size=set["meta"]["size"],
            children=set["meta"]["children"],
            datapoints=set["datapoints"],
            name=set["name"],
            lazy=lazy,
        )

    # Fetch (part of) input or output data for the dataset. If 'lazy' is True, the cache is referenced to see if there is a cached version of the dataset present.
    def _get_data(self, kind, start=0, end=None, otype="raw", silent=True, lazy=True):
        if kind in ["in", "out"]:
            data_dim = None
            arr = None
            if end is None:
                end = self.meta["size"]
            if lazy and exists(f"""src\\datasets\\cache\\{self.name}.npz"""):
                with np.load(
                    f"""src\\datasets\\cache\\{self.name}.npz""", allow_pickle=True
                ) as data:
                    return data[f"{kind}puts"][start:end, :]
            else:
                num_el = end - start
                for i in tqdm(range(num_el), disable=silent):
                    dp = Datapoint.from_json(
                        f"""./src/datapoints/{self.datapoints[start + i]}"""
                    )
                    if data_dim is None:
                        if kind == "out":
                            if (
                                dp.output is None
                                or not "aggregated" in dp.output.keys()
                            ):
                                return None
                            else:
                                data_dim = len(dp.output[otype])
                        else:
                            if dp.input is None:
                                return None
                            else:
                                data_dim = len(dp.input)
                        arr = np.ones([num_el, data_dim])
                    if kind == "out":
                        arr[i, :] = dp.output[otype]
                    else:
                        arr[i, :] = list(dp.input.values())
            return arr

    # Get (part of) the inputs of the datapoints of the dataset
    def get_inputs(self, start=0, end=None, silent=True, lazy=True):
        return self._get_data("in", start=start, end=end, silent=silent, lazy=lazy)

    # Get (part of) the outputs of the datapoints of the dataset.
    def get_outputs(
        self, start=0, end=None, otype="aggregated", silent=True, lazy=True
    ):
        return self._get_data(
            "out", start=start, end=end, otype=otype, silent=silent, lazy=lazy
        )

    # Check if the current dataset has any datapoints that have a corrupted json file in it.
    def is_sane(self):
        corrupted_list = []
        for dp_name in self.datapoints:
            try:
                dp = Datapoint.from_json(f"""./src/datapoints/{dp_name}""")
            except (json.decoder.JSONDecodeError):
                corrupted_list.append(dp_name)
        if len(corrupted_list) == 0:
            return True
        else:
            return False
