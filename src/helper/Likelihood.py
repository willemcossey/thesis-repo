from abc import abstractmethod
from src.helper.SimulationJob import SimulationJob
from math import sqrt
from src.helper.SampleCollection import SampleCollection


class LikeliHood:
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


class SimulationLikelihood(LikeliHood):
    def evaluate(self):
        # generate simulated outcome from parameters
        m = self.parameters[0]
        lmb = self.parameters[1]

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
        sim.run()
        # compare observed data with simulated outcome
        outcome = SampleCollection(sim.result)
        observed = self.data
        # return result
        return outcome.compare(observed)
