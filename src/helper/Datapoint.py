import json
from helper.SimulationJob import SimulationJob
import numpy as np
import hashlib
from threading import Thread


class Datapoint:
    def __init__(self, input_dict, assumptions_dict, output=None, name=None):
        self.input = input_dict
        self.meta = assumptions_dict
        self.output = output
        if name is None:
            self.name = hashlib.md5(
                str(sorted(self.meta.items())).encode("utf-8")
            ).hexdigest()
        else:
            self.name = name
        self.save()

    def save(self):
        t = Thread(target=self._save)
        t.start()
        t.join()
        pass

    def _save(self):
        jstr = self.to_json()
        with open(f"""src/datapoints/{self.name}.json""", mode="w") as file:
            json.dump(jstr, file, indent=1)
            file.flush()
            file.close()
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
        h = 2 / n
        self.output["aggregated"] = list(
            np.histogram(self.output["raw"], n, range=[-1, 1], density=True)[0] * h
        )
        self.save()
        pass

    def to_json(self):
        return self.__dict__

    @staticmethod
    def from_json(filename):
        f = open(filename, "r+")

        try:
            point = json.load(f)
            f.flush()
            f.close()
            return Datapoint(
                point["input"], point["meta"], point["output"], point["name"]
            )
        except json.decoder.JSONDecodeError:
            print(f"Corrupted file at following location: {filename}")
