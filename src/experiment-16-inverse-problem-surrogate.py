import numpy as np
import os.path
from helper.ExperimentVisualizer import ExperimentVisualizer
from helper.InverseProblem import InverseProblem

# from helper.ExperimentVisualizer import ExperimentVisualizer
import time
import argparse
import random

start_minutes = int(time.time() // 60)

parser = argparse.ArgumentParser(
    description="Solve inverse problem with surrogate likelihood."
)

parser.add_argument(
    "filename", metavar="filename", type=str, help="filename of synthetic data file"
)
parser.add_argument(
    "-tl",
    "--true_lambda",
    dest="true_lambda",
    type=float,
    help="true underlying lambda",
)
parser.add_argument(
    "-tm", "--true_m", dest="true_m", type=float, help="true underlying mean opinion"
)
parser.add_argument(
    "-o",
    "--observations",
    dest="observations",
    type=int,
    default=50,
    help="number of observations from synthetic data",
)
parser.add_argument(
    "-w",
    "--noise",
    dest="noise_level",
    type=float,
    default=0.05,
    help="noise level added to synthetic data",
)
parser.add_argument(
    "-s",
    "--samples",
    dest="samples",
    type=int,
    default=3000,
    help="number of resulting samples",
)
parser.add_argument(
    "-b",
    "--burn",
    dest="burn",
    type=int,
    default=1000,
    help="number of burn-in samples",
)
parser.add_argument(
    "-p",
    "--prop",
    dest="proposal",
    type=float,
    default=0.05,
    help="magnitude proposal distribution standard deviation",
)
parser.add_argument(
    "-r",
    "--rng_seed",
    dest="seed",
    type=int,
    default=None,
    help="random number generator seed to be used",
)
parser.add_argument("--show", action="store_true")
parser.add_argument(
    "-loc",
    "--nn-loc",
    dest="nn_loc",
    type=str,
    default=None,
    help="Specify file location of the NN surrogate model to be used.",
)
args = parser.parse_args()

if args.seed is None:
    seed = random.randint(1, 2**32 - 1)
else:
    seed = args.seed

if args.nn_loc is None:
    raise ValueError

random.seed(seed)
np.random.seed(seed)

gamma = 0.01
assert gamma > 0
experiment_assumptions = dict(
    free_parameters={"lmb", "m"},
    theta_bound=lambda g, w: (1 - g) / (1 + abs(w)),
    gamma=gamma,
    lmb_bound=(1 / (3 * gamma) - 2 / 3 + gamma / 3),
    p=lambda w: 1,
    d=lambda w: (1 - w**2),
    nn_loc=args.nn_loc,
)

proposal_step_size = args.proposal

solver_settings = dict(
    num_rounds=args.samples,
    num_burn_in=args.burn,
    initial_sample=dict(lmb=1.5, m=-0.2),
    proposal_std=dict(lmb=proposal_step_size, m=proposal_step_size),
    lh_type="nn",
)

synth_data_file = args.filename

# synth data:
synth_data = np.load(
    os.path.join(
        "experiment-data",
        synth_data_file,
    )
)
underlying_m = args.true_m
underlyng_lmb = args.true_lambda

n_observations = args.observations
observed_data = synth_data[0:n_observations]
# add noise
noise_std = args.noise_level
noisy_observed_data = InverseProblem.add_noise(observed_data, noise_std)

# solve inverse problem for this data
problem = InverseProblem(noisy_observed_data, experiment_assumptions, solver_settings)
samples = problem.solve(silent=True)


output_file = os.path.join(
    "experiment-data",
    f"experiment-16--{synth_data_file}--noise-{noise_std}-n_observations-{n_observations}-num_rounds-{solver_settings['num_rounds']}-burn_in-{solver_settings['num_burn_in']}-proposal--{solver_settings['proposal_std']['lmb']}-{solver_settings['proposal_std']['m']}--initial_sample--{solver_settings['initial_sample']['lmb']}-{solver_settings['initial_sample']['m']}-nn-{solver_settings['lh_type']}-start-{start_minutes}-seed-{seed}",
)

np.savez(output_file, nn=args.nn_loc, lmb=samples["lmb"], m=samples["m"])

#%% plot results

if args.show:
    ExperimentVisualizer.from_samples_file(
        output_file + ".npz",
        solver_settings["num_burn_in"],
        underlyng_lmb,
        underlying_m,
        mode="series",
        block=False,
    )
