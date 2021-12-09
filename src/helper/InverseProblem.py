from src.helper.Distribution import Uniform, Normal
from src.helper.Likelihood import LikeliHood
from tqdm import tqdm


class InverseProblem:
    def __init__(self, observed_data):
        self.observed_data = observed_data
        gamma = 0.01
        assert gamma > 0
        self.experiment_assumptions = dict(
            free_parameters={"lmb", "m"},
            theta_bound=lambda g, w: (1 - g) / (1 + abs(w)),
            gamma=gamma,
            lmb_bound=(1 / (3 * gamma) - 2 / 3 + gamma / 3),
            p=lambda w: 1,
            d=lambda w: (1 - w ** 2),
            t_horiz=200,
            nagents=10000,
        )
        self.prior_dict = dict(
            lmb=Uniform(0, self.experiment_assumptions["lmb_bound"]), m=Uniform(-1, 1)
        )

        # Solve inverse problem by Posterior Sampling using Metropolis Hastings

    def solve(self):

        # initialize parameters
        solver_settings = dict(
            num_rounds=1000,
            num_burn_in=300,
            initial_sample=dict(lmb=0.1, m=0),
            variation=dict(lmb=0.1, m=0.1),
        )

        # establish old sample Prior x LH
        current_sample = solver_settings["initial_sample"]
        current_sample_post_value = self._evaluate_post_value(current_sample)

        resulting_samples = dict(lmb=[current_sample["lmb"]], m=[current_sample["m"]])
        resulting_post_values = [current_sample_post_value]

        for n in tqdm(
            range(0, solver_settings["num_rounds"] + solver_settings["num_burn_in"])
        ):
            proposal = self._propose_new_sample(current_sample, solver_settings)
            current_sample, current_sample_post_value = self._determine_new_sample(
                proposal, current_sample, current_sample_post_value
            )
            resulting_samples["lmb"].append(current_sample["lmb"])
            resulting_samples["m"].append(current_sample["m"])
            resulting_post_values.append(current_sample_post_value)

        return resulting_samples

    def _evaluate_prior(self, sample):
        assert self._sound_sample(sample)
        return self.prior_dict["lmb"].evaluate(sample["lmb"]) * self.prior_dict[
            "m"
        ].evaluate(sample["m"])

    def _propose_new_sample(self, old_sample, solver_settings):
        assert self._sound_sample(old_sample)
        new_sample = old_sample
        for key in old_sample.keys():
            new_sample[key] = (
                new_sample[key] + Normal(0, solver_settings["variation"][key]).sample()
            )
        return new_sample

    def _evaluate_post_value(self, sample):
        assert self._sound_sample(sample)
        return (
            self._evaluate_prior(sample)
            * LikeliHood(
                self.observed_data, sample, self.experiment_assumptions
            ).evaluate()
        )

    def _determine_new_sample(self, new_sample, old_sample, old_sample_post_value):

        new_sample_post_value = self._evaluate_post_value(new_sample)
        ratio = new_sample_post_value / old_sample_post_value
        dec = Uniform().sample()

        if dec >= 1 - ratio:
            return new_sample, new_sample_post_value
        else:
            return old_sample, old_sample_post_value

    def _sound_sample(self, parameters):
        parameter_keys = self.experiment_assumptions["free_parameters"]
        return (
            (isinstance(parameters, dict))
            & (len(parameters) == len(parameter_keys))
            & (set(parameters.keys()) == parameter_keys)
        )
