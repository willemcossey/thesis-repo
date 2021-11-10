from src.helper.Distribution import *
import pytest


# Create fixtures


def example_normal_dist():
    return Normal()


def example_uniform_dist():
    return Uniform()


def all_dist_example_list():
    return [example_normal_dist(), example_uniform_dist()]


class TestDistribution:
    # test what happens if illegitimate inputs

    #     general
    #         neg or zero amount of samples requested
    @pytest.mark.parametrize("distribution", all_dist_example_list())
    def test_sample_negative_amount(self, distribution):
        with pytest.raises(ValueError):
            distribution.sample(amount=-1)

    @pytest.mark.parametrize("distribution", all_dist_example_list())
    def test_sample_zero_amount(self, distribution):
        assert distribution.sample(amount=0) == []

    #         wrong type amount requested
    @pytest.mark.parametrize("distribution", all_dist_example_list())
    def test_sample_string_amount(self, distribution):
        with pytest.raises(TypeError):
            distribution.sample(amount="two")

    #     uniform dist
    #         test smaller upper than lower
    def test_invalid_boundaries(self):
        with pytest.raises(AssertionError):
            Uniform(lower=1, upper=0)

    #     normal dist
    #         wrong type mean
    def test_mean_string_type(self):
        with pytest.raises(AssertionError):
            Normal("zero")

    #         wrong type std
    def test_std_string_type(self):
        with pytest.raises(AssertionError):
            Normal(0, "0.1")

    #         neg std
    def test_negative_standard_deviation(self):
        with pytest.raises(AssertionError):
            Normal(0, -0.1)

    # test what gets returned
    #      general
    #         test if sample returns list
    def test_sample_return_type(self):
        result_sample = example_normal_dist().sample(3)
        assert isinstance(result_sample, list)

    #         test if sample returns right length
    @pytest.mark.parametrize("amount", [1, 10, 10000])
    def test_sample_return_length(self, amount):
        pass

    # truncated normal distribution
    @pytest.mark.parametrize("bounds", [[-1, 1], [-0.1, 0.1], [-5, 5]])
    def test_bounds_respected(self, bounds):
        samples = TruncatedNormal(0.1, 2, bounds).sample(10000)
        assert all([(s <= bounds[1]) | (s >= bounds[0]) for s in samples])
