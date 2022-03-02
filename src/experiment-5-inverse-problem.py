import numpy as np
import os.path
from helper.Distribution import Normal,Uniform
from helper.InverseProblem import InverseProblem
from helper.ExperimentVisualizer import ExperimentVisualizer

import argparse

parser = argparse.ArgumentParser(description='Solve inverse problem.')

parser.add_argument('filename', metavar='filename', type=str, help="filename of synthetic data file")
parser.add_argument("-o","--observations", dest="observations",type=int,default=50,help="number of observations from synthetic data")
parser.add_argument("-w","--noise",dest="noise_level",type=float,default=.05,help="noise level added to synthetic data")
parser.add_argument("-s","--samples", dest="samples",type=int,default=10,help="number of resulting samples")
parser.add_argument("-b","--burn", dest="burn",type=int,default=10,help="number of burn-in samples")
parser.add_argument("-p","--prop", dest="proposal",type=float,default=.05,help="magnitude proposal distribution standard deviation")
parser.add_argument("-t", "--t_horiz", dest="t_horiz", type=int,default = 100, help="time horizon")
parser.add_argument("-n", "--nagents", dest="nagents", type=int,default = 1000, help="number of agents")

args = parser.parse_args()

gamma = 0.01
assert gamma > 0
experiment_assumptions = dict(
    free_parameters={"lmb", "m"},
    theta_bound=lambda g, w: (1 - g) / (1 + abs(w)),
    gamma=gamma,
    lmb_bound=(1 / (3 * gamma) - 2 / 3 + gamma / 3),
    p=lambda w: 1,
    d=lambda w: (1 - w ** 2),
    t_horiz=args.t_horiz,
    nagents=args.nagents,
)

proposal_step_size = args.proposal

solver_settings = dict(
    num_rounds=args.samples,
    num_burn_in=args.burn,
    initial_sample=dict(lmb=0.5, m=-0.5),
    proposal_std=dict(lmb=proposal_step_size, m=proposal_step_size),
)

synth_data_file = args.filename

# synth data:
synth_data = np.load(
    os.path.join(
        "experiment-data",
        synth_data_file,
    )
)
underlying_m = -0.5
underlyng_lmb = 0.5

n_observations = args.observations
observed_data = synth_data[0:n_observations]
# add noise
noise_std = args.noise_level
noisy_observed_data = InverseProblem.add_noise(observed_data, noise_std)

# solve inverse problem for this data
problem = InverseProblem(noisy_observed_data, experiment_assumptions)
samples = problem.solve(solver_settings)


output_file = os.path.join(
    "experiment-data",
    f"experiment-5-inverse-problem-from--{synth_data_file}--noise_std-{noise_std}-n_observations-{n_observations}-num_rounds-{solver_settings['num_rounds']}-burn_in-{solver_settings['num_burn_in']}-proposal_std--{solver_settings['proposal_std']['lmb']}-{solver_settings['proposal_std']['m']}--initial_sample--{solver_settings['initial_sample']['lmb']}-{solver_settings['initial_sample']['m']}--t_horiz-{experiment_assumptions['t_horiz']}-nagents-{experiment_assumptions['nagents']}",
)

np.savez(output_file, lmb=samples["lmb"], m=samples["m"])


# plot results
ExperimentVisualizer.from_samples_file(
    output_file + ".npz", solver_settings["num_burn_in"], underlyng_lmb, underlying_m, mode='series')



