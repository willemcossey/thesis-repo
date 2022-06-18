import os.path as pth
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from os import path

mpl.style.use(path.join("src", "grayscale_adjusted.mplstyle"))
mpl.rcParams["xtick.labelsize"] = 10

# load an experiment from experiment-data


class ExperimentVisualizer:
    def __init__(self):
        pass

    @staticmethod
    def from_file(filename):
        try:
            filepath = pth.join("experiment-data", filename)
            experiment_data = np.load(filepath)
            plt = ExperimentVisualizer.from_array(experiment_data)
            return plt
        except IOError:
            print(f"Not found: {filepath}")

    @staticmethod
    def from_array(arr, block=True):
        plt.figure()
        counts, bins = np.histogram(arr, bins=np.linspace(-1, 1, 200))
        bins = 0.5 * (bins[:-1] + bins[1:])
        plt.bar(x=bins, height=counts, width=2 / len(bins))
        plt.xlabel("Opinion $w$")
        plt.ylabel("Count")
        plt.show(block=block)
        return plt

    @staticmethod
    def from_samples_file(
        filename, burn_in, lmb, m, mode="hist", block=True, title=True
    ):
        num_burn_in = burn_in
        try:
            filepath = filename
            print(f"Plotting parameter estimates from {filepath}")
            experiment_data = np.load(filepath)
            if mode == "hist":
                f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
                ax1.hist(
                    experiment_data["lmb"][num_burn_in:],
                    bins=np.linspace(
                        0,
                        max(max(experiment_data["lmb"][num_burn_in:]), 2, 2 * lmb),
                        200,
                    ),
                )
                ax1.set_xlabel("$\lambda$")
                ax1.set_ylabel("Count")
                ax2.hist(
                    experiment_data["m"][num_burn_in:], bins=np.linspace(-1, 1, 200)
                )
                ax2.set_xlabel("$m$")
                ax2.set_ylabel("Count")
                if title:
                    f.suptitle(
                        f"Ensemble averages: $\lambda$ = {np.mean(experiment_data['lmb'][num_burn_in:]): .2f}, $m$ = {np.mean(experiment_data['m'][num_burn_in:]): .2f}"
                    )
                plt.show(block=block)
            if mode == "series":
                f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                ax1.plot(
                    range(num_burn_in, len(experiment_data["lmb"])),
                    experiment_data["lmb"][num_burn_in:],
                )
                ax1.plot(
                    range(0, num_burn_in),
                    experiment_data["lmb"][0:num_burn_in],
                    "-.",
                    label="burn-in",
                )
                ax1.set_xlabel("index")
                ax1.set_ylabel("$\lambda$")
                ax2.plot(
                    range(num_burn_in, len(experiment_data["m"])),
                    experiment_data["m"][num_burn_in:],
                )
                ax2.plot(
                    range(0, num_burn_in), experiment_data["m"][0:num_burn_in], "-."
                )
                ax2.plot(
                    range(num_burn_in, len(experiment_data["m"])),
                    experiment_data["m"][num_burn_in:],
                )
                ax2.set_xlabel("index")
                ax2.set_ylabel("$m$")
                if title:
                    f.suptitle(
                        f"Ensemble averages: $\lambda$ = {np.mean(experiment_data['lmb'][num_burn_in:]): .2f}, $m$ = {np.mean(experiment_data['m'][num_burn_in:]): .2f}"
                    )
                ax1.legend()
                plt.show(block=block)

        except IOError:
            print(f"Not found: {filename}")

        return plt
