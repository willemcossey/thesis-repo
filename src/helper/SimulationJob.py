from random import randrange
from helper.Distribution import Uniform, TruncatedNormal
from helper.OpinionModel import OpinionModel
from tqdm import tqdm


class SimulationJob:
    def __init__(
        self,
        gamma,
        theta_std,
        theta_bound,
        p,
        d,
        mean_opinion: [float, int] = 0,
        t_end=1,
        n_samples=2000,
        uniform_theta=False,
    ):
        self.model = OpinionModel(gamma, p, d, theta_std, theta_bound, uniform_theta)
        # self.init_dist = TruncatedNormal(mean_opinion, 0.5, [-1, 1]) #wrong mean
        self.init_dist = Uniform(
            -1 + max(2 * mean_opinion, 0), 1 + min(0, 2 * mean_opinion)
        )
        self.time_horizon = t_end
        if n_samples > 1:
            self.n_samples = n_samples
        else:
            raise ValueError("Number of samples should be at least 2")
        self.result = None
        self.hide_progress = False

    def run(self, init_samples=None):
        # based on P&T p.135 bottom of page

        # set time to simulate

        # generate initial distribution samples
        if init_samples is None:
            samples = self.init_dist.sample(self.n_samples)
        else:
            samples = init_samples

        # set timestep, n_iterations and current opinion dist to initial
        n_steps = self.time_horizon * self.n_samples
        # for counter = 0, n_iterations
        it = 0
        for it in tqdm(range(n_steps), disable=self.hide_progress):
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
