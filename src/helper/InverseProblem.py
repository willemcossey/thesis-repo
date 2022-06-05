from helper.Distribution import Uniform, Normal
from helper.Likelihood import SimulationLikelihood, NNLikelihood
from tqdm import tqdm
import numpy as np


class InverseProblem:
    def __init__(self, observed_data, experiment_assumptions, solver_settings):
        self.observed_data = observed_data
        self.experiment_assumptions = experiment_assumptions
        self.prior_dict = dict(
            lmb=Uniform(0, self.experiment_assumptions["lmb_bound"]), m=Uniform(-1, 1)
        )
        self.solver_settings = solver_settings

        # Solve inverse problem by Posterior Sampling using Metropolis Hastings

    def solve(self):

        # establish old sample Prior x LH
        current_sample = self.solver_settings["initial_sample"]
        current_sample_post_value = self._evaluate_post_value(current_sample)

        resulting_samples = dict(lmb=[current_sample["lmb"]], m=[current_sample["m"]])
        resulting_post_values = [current_sample_post_value]

        for n in tqdm(
            range(
                0,
                self.solver_settings["num_rounds"]
                + self.solver_settings["num_burn_in"],
            )
        ):
            proposal = self._propose_new_sample(current_sample)
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

    def _propose_new_sample(self, old_sample):
        assert self._sound_sample(old_sample)
        new_sample = old_sample.copy()
        for key in old_sample.keys():
            new_sample[key] = (
                new_sample[key]
                + Normal(0, self.solver_settings["proposal_std"][key]).sample()[0]
            )
        return new_sample

    def _evaluate_post_value(self, sample):
        assert self._sound_sample(sample)
        prior = self._evaluate_prior(sample)
        if prior == 0:
            return prior
        else:

            if self.solver_settings["lh_type"] == "nn":
                lh = NNLikelihood(
                    self.observed_data, sample, self.experiment_assumptions
                )
            else:
                lh = SimulationLikelihood(
                    self.observed_data, sample, self.experiment_assumptions
                )
            lh_value = lh.evaluate()
            return prior * lh_value

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

    @staticmethod
    def add_noise(data, noise_std):
        data = np.array(data)
        noise_dist = Normal(0, noise_std)
        noise = np.array(noise_dist.sample(amount=len(data)))
        noisy_data = data + noise
        for i in range(0, len(noisy_data)):
            while abs(noisy_data[i]) > 1:
                noisy_data[i] = data[i] + noise_dist.sample()
        return noisy_data
