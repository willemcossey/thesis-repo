from src.helper.OpinionModel import *
import pytest


def diff_type_arg_list():
    return ["0.1", [0.1], set(), tuple()]


class TestOpinionModel:
    @pytest.fixture
    def default_opinion_model(self):
        return OpinionModel()

    # test constructor
    # how are invalid inputs handled
    # gamma outside domain
    @pytest.mark.parametrize("value", [-0.1, 0.55])
    def test_gamma_invalid_value(self, value):
        with pytest.raises(AssertionError):
            OpinionModel(gamma=value)

    # gamma invalid type
    @pytest.mark.parametrize("diff_type_arg", diff_type_arg_list())
    def test_gamma_invalid_type(self, diff_type_arg):
        with pytest.raises(TypeError):
            OpinionModel(gamma=diff_type_arg)

    # theta invalid type
    @pytest.mark.parametrize("diff_type_arg", diff_type_arg_list())
    def test_theta_invalid_type(self, diff_type_arg):
        with pytest.raises(TypeError):
            OpinionModel(theta_dist=diff_type_arg)

    # p invalid type
    @pytest.mark.parametrize("diff_type_arg", diff_type_arg_list())
    def test_p_invalid_type(self, diff_type_arg):
        with pytest.raises(TypeError):
            OpinionModel(p=diff_type_arg)

    # p value greater than 1 over domain [-1,1]
    # skipped since the only way to check would be v expensive
    # def test_p_invalid_behaviour(self):
    #     pass

    # d invalid type
    @pytest.mark.parametrize("diff_type_arg", diff_type_arg_list())
    def test_d_invalid_type(self, diff_type_arg):
        with pytest.raises(TypeError):
            OpinionModel(d=diff_type_arg)

    # d value greater than 1 over domain [-1,1]
    # skipped since the only way to check would be v expensive
    # def test_d_invalid_behaviour(self):
    #     pass

    # test apply_operator

    # how are invalid inputs handled
    # invalid sample type
    def test_input_sample_type(self, default_opinion_model):
        with pytest.raises(TypeError):
            OpinionModel.apply_operator(default_opinion_model, set([-0.5, 0.5]))

    # invalid sample length
    def test_input_sample_length(self, default_opinion_model):
        with pytest.raises(AssertionError):
            OpinionModel.apply_operator(default_opinion_model, [-0.5, 0, 0.5])

    # invalid sample value
    def test_input_sample_value(self, default_opinion_model):
        with pytest.raises(AssertionError):
            OpinionModel.apply_operator(default_opinion_model, [-1.5, 1.5])

    # does the method satisfy expected behaviour

    # TODO can this be parametrized to be more significant ?

    # check result sample type
    def test_return_sample_type(self, default_opinion_model):
        result = OpinionModel.apply_operator(default_opinion_model, [-0.5, 0.5])
        assert isinstance(result, list)

    # check result sample length
    def test_return_sample_length(self, default_opinion_model):
        result = OpinionModel.apply_operator(default_opinion_model, [-0.5, 0.5])
        assert len(result) == 2

    # check result sample value
    def test_return_sample_value(self, default_opinion_model):
        result = OpinionModel.apply_operator(default_opinion_model, [-1, 1])
        assert ((result[0] <= 1) & (result[0] >= -1)) & (
            (result[1] <= 1) & (result[1] >= -1)
        )
