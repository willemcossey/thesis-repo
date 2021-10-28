from abc import ABC, abstractmethod
from numpy import random


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
        assert upper >= lower
        self.lower = lower
        self.upper = upper

    def sample(self, amount: int = 1) -> list[float]:
        return list(random.uniform(self.lower, self.upper, amount))


class Normal(Distribution):
    def __init__(self, mean=0, std=1):
        Distribution.__init__(self)
        assert isinstance(mean, (float, int))
        assert isinstance(std, (float, int))
        assert std > 0
        self.mean = mean
        self.std = std

    def sample(self, amount: int = 1) -> list[float]:
        return list(random.normal(self.mean, self.std, amount))
