from helper.Distribution import Normal, TruncatedNormal, Uniform
from math import sqrt


# As described in P&T p. 227


class OpinionModel:
    def __init__(
        self,
        gamma: float = 0.25,
        p: callable = lambda x: 1,
        d: callable = lambda x: 1,
        theta_std: float = 0.05,
        theta_bnd: callable = lambda gamma, w: (1 - gamma) / (1 + abs(w)),
        uniform_theta: bool = False,
    ):
        if not isinstance(gamma, (float, int)):
            raise TypeError
        if not callable(p):
            raise TypeError
        if not callable(d):
            raise TypeError
        assert (gamma >= 0) & (gamma <= 0.5)
        assert isinstance(theta_std, float)
        assert callable(theta_bnd)
        self.gamma = gamma
        self.P = p
        self.D = d
        self.theta_std = theta_std
        self.theta_bound = theta_bnd
        self.uniform_theta = uniform_theta

    def apply_operator(self, two_samples: list[(float, int)]) -> list:
        assert len(two_samples) == 2
        assert ((two_samples[0] <= 1) & (two_samples[0] >= -1)) & (
            (two_samples[1] <= 1) & (two_samples[1] >= -1)
        )

        new_samples = two_samples.copy()
        diff = new_samples[0] - new_samples[1]

        theta_samples = [None] * 2
        compromise = [None] * 2
        diffusion = [None] * 2

        # first agent
        theta_support = self.theta_bound(self.gamma, new_samples[0])
        theta_dist = self.get_theta_dist(theta_support)

        theta_samples[0] = theta_dist.sample()[0]
        compromise[0] = -1 * self.gamma * self.P(abs(new_samples[0])) * diff
        diffusion[0] = theta_samples[0] * self.D(abs(new_samples[0]))

        new_samples[0] = new_samples[0] + compromise[0] + diffusion[0]

        # second agent
        theta_support = self.theta_bound(self.gamma, new_samples[1])
        theta_dist = self.get_theta_dist(theta_support)

        theta_samples[1] = theta_dist.sample()[0]
        compromise[1] = self.gamma * self.P(abs(new_samples[1])) * diff
        diffusion[1] = theta_samples[1] * self.D(abs(new_samples[1]))

        new_samples[1] = new_samples[1] + compromise[1] + diffusion[1]

        assert (new_samples[0] >= -1) & (new_samples[0] <= 1)
        assert (new_samples[1] >= -1) & (new_samples[1] <= 1)

        return new_samples

    def get_theta_dist(self, support):
        if support is None:
            return Normal(0, self.theta_std)
        else:
            if self.uniform_theta:
                b = sqrt(3) * self.theta_std
                assert b <= support
                return Uniform(-b, b)
            else:
                return TruncatedNormal(0, self.theta_std, [-support, support])
