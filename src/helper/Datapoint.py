import json
from helper.SimulationJob import SimulationJob
import numpy as np


class Datapoint:
    def __init__(self, input_dict, assumptions_dict, output=None, name=None):
        self.input = input_dict
        self.meta = assumptions_dict
        self.output = output
        if name is None:
            self.name = hash(tuple(str(sorted(self.meta.items()))))
        else:
            self.name = name
        self.save()

    def save(self):
        file = open(f"""src\\datapoints\\{self.name}.json""", mode="w")
        json.dump(self.to_json(), file, indent=1)
        pass

    def compute_output(self, write=True):
        sim_obj = SimulationJob(
            self.meta["gamma"],
            self.meta["theta_std"],
            eval(self.meta["theta_bound"]),
            eval(self.meta["p"]),
            eval(self.meta["d"]),
            self.meta["mean_opinion"],
            self.meta["t_end"],
            self.meta["n_samples"],
            self.meta["uniform_theta"],
        )
        sim_obj.run()

        self.output = {"raw": sim_obj.result}
        self.save()
        pass

    def compute_aggregated_output(self, n):
        self.output["aggregated"] = list(
            np.histogram(self.output["raw"], n, range=[-1, 1], density=True)[0]
        )
        self.save()
        pass

    def to_json(self):
        return self.__dict__

    @staticmethod
    def from_json(filename):
        f = open(filename)
        point = json.load(f)
        return Datapoint(point["input"], point["meta"], point["output"], point["name"])
