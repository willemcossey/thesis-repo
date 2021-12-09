from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import truncnorm
from itertools import chain


class Distribution(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, amount: int = 1) -> list[float]:
        pass


class Uniform(Distribution):
    def __init__(self, lower: int = 0, upper: int = 1):
        Distribution.__init__(self)
        assert upper > lower
        self.lower = lower
        self.upper = upper

    def sample(self, amount: int = 1) -> list[float]:
        return list(np.random.uniform(self.lower, self.upper, amount))

    def evaluate(self, p):
        return 1 / (self.upper - self.lower)


class Normal(Distribution):
    def __init__(self, mean=0, std=1):
        Distribution.__init__(self)
        assert isinstance(mean, (float, int))
        assert isinstance(std, (float, int))
        assert std > 0
        self.mean = mean
        self.std = std

    def sample(self, amount: int = 1) -> list[float]:
        return list(
            chain.from_iterable(list(np.random.normal(self.mean, self.std, amount)))
        )


class TruncatedNormal(Distribution):
    def __init__(self, mean=0, std=1, bounds=None):
        Distribution.__init__(self)
        if bounds is None:
            bounds = [-1, 1]
        assert isinstance(mean, (float, int))
        assert isinstance(std, (float, int))
        assert std > 0
        assert isinstance(bounds, list)
        assert len(bounds) == 2
        assert bounds[1] >= bounds[0]
        self.mean = mean
        self.std = std
        self.bounds = bounds

    def sample(self, amount: int = 1) -> list[float]:
        a = (self.bounds[0] - self.mean) / self.std * np.ones([amount, 1])
        b = (self.bounds[1] - self.mean) / self.std * np.ones([amount, 1])
        loc = self.mean * np.ones([amount, 1]) * self.std

        samples_std_dist = truncnorm.rvs(a, b)
        samples_result = (samples_std_dist * self.std) + self.mean

        # return samples as list
        sample_lst = (
            [float(samples_result)]
            if amount == 1
            else list(chain.from_iterable(np.transpose(samples_result).tolist()))
        )

        return sample_lst
