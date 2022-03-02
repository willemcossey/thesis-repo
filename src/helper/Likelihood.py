from abc import abstractmethod
from helper.SimulationJob import SimulationJob
from math import sqrt
from helper.SampleCollection import SampleCollection


class _LikeliHood:
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


class SimulationLikelihood(_LikeliHood):
    def __init__(
        self,
        data,
        parameters,
        experiment_assumptions,
    ):
        _LikeliHood.__init__(self, data, parameters, experiment_assumptions)

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
        return outcome.compare(observed)
