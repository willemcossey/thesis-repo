from abc import ABC, abstractmethod
from numpy import random


class Distribution(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def sample(self, amount: int) -> list[float]:
        pass


class Uniform(Distribution):

    def __init__(self, lower=0, upper=1):
        Distribution.__init__(self)
        self.lower = lower
        self.upper = upper

    def sample(self, amount=1):
        return list(random.uniform(self.lower, self.upper, amount))

class Normal(Distribution):

    def __init__(self, mean = 0, std = 1):
        Distribution.__init__(self)
        self.mean = mean
        self.std = std

    def sample(self, amount=1):
        return random.normal(self.mean, self.std, amount)