import numpy as np
import os.path
from src.helper.Distribution import Normal
from src.helper.InverseProblem import InverseProblem
from src.helper.ExperimentVisualizer import ExperimentVisualizer

gamma = 0.01
assert gamma > 0
experiment_assumptions = dict(
    free_parameters={"lmb", "m"},
    theta_bound=lambda g, w: (1 - g) / (1 + abs(w)),
    gamma=gamma,
    lmb_bound=(1 / (3 * gamma) - 2 / 3 + gamma / 3),
    p=lambda w: 1,
    d=lambda w: (1 - w ** 2),
    t_horiz=100,
    nagents=1000,
)

solver_settings = dict(
    num_rounds=12000,
    num_burn_in=300,
    initial_sample=dict(lmb=0.2, m=-0.3),
    proposal_std=dict(lmb=0.05, m=0.05),
)

synth_data_file = "synth-data-lmb-0.5-m--0.5-t_horiz-200-nagents-100000.npy"

# synth data:
synth_data = np.load(
    os.path.join(
        "C:\\Users\\wille",
        "thesis-repo\\src",
        "experiment-data",
        synth_data_file,
    )
)
underlying_m = -0.5
underlyng_lmb = 0.5

n_observations = 1000
observed_data = synth_data[0:n_observations]
# add noise
noise_std = 0.05
noise = Normal(0, noise_std).sample(n_observations)
observed_data = observed_data + noise

# solve inverse problem for this data
problem = InverseProblem(observed_data, experiment_assumptions)
samples = problem.solve(solver_settings)


output_file = os.path.join(
    "experiment-data",
    f"experiment-5-inverse-problem-from--{synth_data_file}--noise_std-{noise_std}-n_observations-{n_observations}-num_rounds-{solver_settings['num_rounds']}-burn_in-{solver_settings['num_burn_in']}-proposal_std--{solver_settings['proposal_std']['lmb']}-{solver_settings['proposal_std']['m']}--initial_sample--{solver_settings['initial_sample']['lmb']}-{solver_settings['initial_sample']['m']}--t_horiz-{experiment_assumptions['t_horiz']}-nagents-{experiment_assumptions['nagents']}",
)

np.savez(output_file, lmb=samples["lmb"], m=samples["m"])


# plot results
ExperimentVisualizer.from_samples_file(
    output_file + ".npz", solver_settings["num_burn_in"], underlyng_lmb, underlying_m
)
