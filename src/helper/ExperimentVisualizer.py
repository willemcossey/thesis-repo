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
