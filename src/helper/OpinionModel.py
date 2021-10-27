from helper.Distribution import Distribution, Normal
import numpy as np

# As described in P&T p. 227


class OpinionModel:
    def __init__(
        self, gamma=0.25, theta_dist=Normal(0, 0.05), p=lambda x: 1, d=lambda x: 1
    ):
        assert (gamma >= 0) & (gamma <= 0.5)
        assert issubclass(type(theta_dist), Distribution)
        assert callable(p)
        assert callable(d)
        self.gamma = gamma
        self.Theta = theta_dist
        self.P = p
        self.D = d

    def apply_operator(self, two_samples) -> list:
        assert type(two_samples) == list
        assert len(two_samples) == 2

        new_samples = two_samples.copy()
        diff = new_samples[0] - new_samples[1]

        two_theta = self.Theta.sample(2)

        new_samples[0] = (
            new_samples[0]
            - self.gamma * self.P(abs(new_samples[0])) * diff
            + two_theta[0] * self.D(abs(new_samples[0]))
        )
        new_samples[1] = (
            new_samples[1]
            - self.gamma * self.P(abs(new_samples[1])) * diff
            + two_theta[1] * self.D(abs(new_samples[1]))
        )

        #check if new samples not nan and \in [-1,1] (apply Beta_int)

        new_samples[0] = new_samples[0] if ((new_samples[0] >= -1) & (new_samples[0] <= 1)) else two_samples[0].copy()
        new_samples[1] = new_samples[1] if ((new_samples[1] >= -1) & (new_samples[1] <= 1)) else two_samples[1].copy()

        assert not np.isnan(new_samples[0])
        assert not np.isnan(new_samples[1])

        return new_samples
