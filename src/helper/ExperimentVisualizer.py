import os.path as pth
import numpy as np
import matplotlib.pyplot as plt


# load an experiment from experiment-data


class ExperimentVisualizer:
    def __init__(self):
        pass

    @staticmethod
    def from_file(filename):
        try:
            filepath = pth.join("../experiment-data", filename)
            experiment_data = np.load(filepath)
            ExperimentVisualizer.from_array(experiment_data)
        except IOError:
            print(IOError)
            # print("This experiment hasn't been run")

        return None

    @staticmethod
    def from_array(arr):
        plt.figure()
        counts, bins = np.histogram(arr, bins=np.linspace(-1, 1, 200))
        bins = 0.5 * (bins[:-1] + bins[1:])
        plt.bar(x=bins, height=counts, width=2 / len(bins))
        plt.xlabel("Opinion []")
        plt.ylabel("Count []")
        plt.show(block=True)

    @staticmethod
    def from_samples_file(filename, burn_in, lmb, m):
        num_burn_in = burn_in
        try:
            filepath = pth.join("../experiment-data", filename)
            experiment_data = np.load(filepath)
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
            counts, bins = np.histogram(
                experiment_data["lmb"][num_burn_in:], bins=np.linspace(0, 2 * lmb, 200)
            )
            bins = 0.5 * (bins[:-1] + bins[1:])
            ax1.bar(x=bins, height=counts, width=2 / len(bins))
            ax1.set_xlabel("lambda []")
            ax1.set_ylabel("Count []")
            counts, bins = np.histogram(
                experiment_data["m"][num_burn_in:], bins=np.linspace(-1, 1, 200)
            )
            bins = 0.5 * (bins[:-1] + bins[1:])
            ax2.bar(x=bins, height=counts, width=2 / len(bins))
            ax2.set_xlabel("mean opinion []")
            ax2.set_ylabel("Count []")
            f.suptitle(
                f"Estimated parameters for data with underlying lambda = {lmb} and m = {m}"
            )

        except IOError:
            print(IOError)
