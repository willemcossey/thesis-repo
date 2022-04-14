from helper.Distribution import Uniform
from helper.Datapoint import Datapoint
import numpy as np
from math import sqrt
import json
import hashlib
from scipy.stats.qmc import Sobol, Halton


class Dataset:
    def __init__(
        self, rng, bounds, assumptions, children=None, size=1, datapoints=[], name=None
    ):
        self.meta = {
            "rng": rng,
            "domain_bounds": bounds,
            "size": size,
            "children": [],
            "experiment_assumptions": assumptions,
        }
        self.datapoints = datapoints
        if children is not None:
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

    def get_size(self):
        return len(self.datapoints)

    def generate_input(self):
        if self.meta["rng"] == "sobol":
            gen = Sobol(len(self.meta["domain_bounds"].keys()))
            points = gen.random(self.size)
            inp = np.transpose(
                np.array(
                    [
                        points[:, i]
                        * (
                            self.meta["domain_bounds"][i][1]
                            - self.meta["domain_bounds"][i][0]
                        )
                        + self.meta["domain_bounds"][i][0]
                        for i in self.meta["domain_bounds"].keys()
                    ]
                )
            )
        if self.meta["rng"] == "halton":
            gen = Halton(len(self.meta["domain_bounds"].keys()))
            points = gen.random(self.size)
            inp = np.transpose(
                np.array(
                    [
                        points[:, i]
                        * (
                            self.meta["domain_bounds"][i][1]
                            - self.meta["domain_bounds"][i][0]
                        )
                        + self.meta["domain_bounds"][i][0]
                        for i in self.meta["domain_bounds"].keys()
                    ]
                )
            )
        else:
            inp = np.array(
                [
                    Uniform(
                        self.meta["domain_bounds"][i][0],
                        self.meta["domain_bounds"][i][1],
                    ).sample(self.meta["size"])
                    for i in self.meta["domain_bounds"].keys()
                ]
            )
        for n in range(self.meta["size"]):
            lmb = float(inp[0, n])
            m = float(inp[1, n])
            theta_std = sqrt(self.meta["experiment_assumptions"]["gamma"] * lmb)
            ass = {
                **self.meta["experiment_assumptions"],
                **{"lmb": lmb, "mean_opinion": m, "theta_std": theta_std},
            }
            dp = Datapoint(
                dict(zip(self.meta["domain_bounds"].keys(), inp[:, n].tolist())), ass
            )
            self.datapoints.append(f"""{dp.name}.json""")
        self.meta["size"] = self.get_size()
        pass

    def import_datasets(self, child_name_array):
        # load datasets
        child_array = []
        for name in child_name_array:
            child_array.append(Dataset.from_json(f"""src\\datasets\\{name}"""))
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

    def compute_output(self):
        for dp_name in self.datapoints:
            dp = Datapoint.from_json(f"""src\\datapoints\\{dp_name}""")
            if (dp.output is None) or (dp.output["raw"] is None):
                dp.compute_output()
        pass

    def save(self):
        file = open(f"""src\\datasets\\{self.name}.json""", mode="w")
        json.dump(self.to_json(), file, indent=1)
        file.flush()
        file.close()
        pass

    def compute_aggregated_output(self, n):
        # Compute the effective 'histogram' of the solution over n equispaced intervals.
        for dp_name in self.datapoints:
            try:
                dp = Datapoint.from_json(f"""src\\datapoints\\{dp_name}""")
                if dp.output is None:
                    dp.compute_output()
                if "aggregated" not in dp.output or len(dp.output["aggregated"]) != n:
                    dp.compute_aggregated_output(n)
            except (json.decoder.JSONDecodeError):
                print(f"Something's wrong with this json file: {dp_name}")
        pass

    def to_json(self):
        return self.__dict__

    @staticmethod
    def from_json(filename):
        f = open(filename)
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
        )

    def get_inputs(self, start=0, end=None):
        input_dim = None
        arr = None
        if end is None:
            end = self.meta["size"]
        num_el = end - start
        for i in range(num_el):
            dp = Datapoint.from_json(f"""src\\datapoints\\{self.datapoints[start+i]}""")
            if input_dim is None:
                input_dim = len(dp.input)
                arr = np.ones([num_el, input_dim])
            arr[i, :] = list(dp.input.values())
        return arr

    def get_outputs(self, start=0, end=None, type="raw"):
        output_dim = None
        arr = None
        if end is None:
            end = self.meta["size"]
        num_el = end - start
        for i in range(num_el):
            dp = Datapoint.from_json(f"""src\\datapoints\\{self.datapoints[start+i]}""")
            if output_dim is None:
                output_dim = len(dp.output[type])
                num_el = end - start
                arr = np.ones([num_el, output_dim])
            arr[i, :] = dp.output[type]
        return arr

    def is_sane(self):
        corrupted_list = []
        for dp_name in self.datapoints:
            try:
                dp = Datapoint.from_json(f"""src\\datapoints\\{dp_name}""")
            except (json.decoder.JSONDecodeError):
                corrupted_list.append(dp_name)
        if len(corrupted_list) == 0:
            return True
        else:
            return False
