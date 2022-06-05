from abc import abstractmethod
from helper.SimulationJob import SimulationJob
from math import sqrt
from helper.SampleCollection import SampleCollection
import torch
import numpy as np


class _Likelihood:
    def __init__(
        self,
        data,
        parameters,
        experiment_assumptions,
    ):
        self.data = data
        self.parameters = parameters
        self.experiment_assumptions = experiment_assumptions

    @abstractmethod
    def evaluate(self):
        pass


class SimulationLikelihood(_Likelihood):
    def __init__(
        self,
        data,
        parameters,
        experiment_assumptions,
    ):
        _Likelihood.__init__(self, data, parameters, experiment_assumptions)

    def evaluate(self):
        # generate simulated outcome from parameters
        m = self.parameters["m"]
        lmb = self.parameters["lmb"]

        gamma = self.experiment_assumptions["gamma"]
        theta_std = sqrt(gamma * lmb)

        sim = SimulationJob(
            gamma,
            theta_std,
            self.experiment_assumptions["theta_bound"],
            self.experiment_assumptions["p"],
            self.experiment_assumptions["d"],
            m,
            self.experiment_assumptions["t_horiz"],
            self.experiment_assumptions["nagents"],
            uniform_theta=True,
        )
        sim.hide_progress = True
        sim.run()
        # compare observed data with simulated outcome
        outcome = SampleCollection(sim.result)
        observed = self.data
        # return result
        return outcome.compare_with_sample_collection(observed)


# experiment assumptions contain location of nn model and if batch prediciton should be applied or not.
class NNLikelihood(_Likelihood):
    def __init__(
        self,
        data,
        parameters,
        experiment_assumptions,
    ):
        _Likelihood.__init__(self, data, parameters, experiment_assumptions)

    def evaluate(self):
        # generate simulated outcome from parameters
        m = self.parameters["m"]
        lmb = self.parameters["lmb"]

        # load nn model
        nn_loc = self.experiment_assumptions["nn_loc"]
        model = torch.load(nn_loc)
        model.eval()

        # evaluate outcome
        x = torch.tensor([lmb, m])
        sim = model(x)

        # compare observed data with simulated outcome
        outcome = np.array(sim.detach())
        observed = SampleCollection(self.data)
        # return result
        return observed.compare_with_hist(outcome)
