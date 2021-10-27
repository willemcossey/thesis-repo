from random import randrange
from helper.Distribution import Uniform
from helper.OpinionModel import OpinionModel


class SimulationJob:
    def __init__(self, gamma, theta, p, d, t_end=1, n_samples=2000):
        self.model = OpinionModel(gamma, theta, p, d)
        self.init_dist = Uniform(lower=-1, upper=1)
        self.time_horizon = t_end
        if n_samples > 1:
            self.n_samples = n_samples
        else:
            raise ValueError("Number of samples should be at least 2")
        self.result = None

    def run(self):
        # based on P&T p.135 bottom of page

        # set time to simulate

        # generate initial distribution samples
        samples = self.init_dist.sample(self.n_samples)

        # set timestep, n_iterations and current opinion dist to initial
        n_steps = self.time_horizon * self.n_samples
        # for counter = 0, n_iterations
        it = 0
        while it < n_steps:
            # select random pair (non-local)

            bob = randrange(0, self.n_samples)
            alice = randrange(0, self.n_samples)
            if bob == alice:
                alice = randrange(0, self.n_samples)
            samples[alice], samples[bob] = self.model.apply_operator(
                [samples[alice], samples[bob]]
            )
            it += 1
        self.result = samples
