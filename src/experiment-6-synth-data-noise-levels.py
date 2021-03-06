from src.helper.InverseProblem import InverseProblem
import numpy as np
from os.path import join
from helper.ExperimentVisualizer import ExperimentVisualizer
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

mpl.style.use(os.path.join("src", "grayscale_adjusted.mplstyle"))

synth_data_file = "synth-data-lmb-1-m--0.1-t_horiz-200-nagents-100000.npy"

# synth data:
synth_data = np.load(
    os.path.join(
        "src",
        "experiment-data",
        synth_data_file,
    )
)

for noise_std in np.linspace(0, 2, 11):
    noisy_data = synth_data
    if noise_std != 0:
        noisy_data = InverseProblem.add_noise(synth_data, noise_std)
    np.save(
        join(
            "src",
            "experiment-data",
            f"experiment-6-noise_std-{round(noise_std,1)}-to-{synth_data_file}-",
        ),
        noisy_data,
    )

    p = ExperimentVisualizer.from_array(noisy_data, block=False)
    p.savefig(
        os.path.join(
            "src",
            f"experiment-data/add-noise-validation/experiment-6-noise_std-{noise_std}-to-{synth_data_file}-.eps",
        )
    )
    # p.show()
    p.close()
