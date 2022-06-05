from scipy.stats import ks_2samp, ks_1samp
from math import floor
import numpy as np


class SampleCollection:
    def __init__(self, lst: list):
        # check if list not nested
        self.members = lst

    # Takes in two collections of samples and returns the probability
    # that both collections stem from the same distribution.
    def compare_with_sample_collection(self, samples, metric="KS"):
        if metric == "KS":
            stat, pval = ks_2samp(self.members, other)
            return pval
        else:
            raise ValueError(
                "This metric is currently not available. Use 'KS' instead."
            )

    def compare_with_hist(self, hist, metric="KS"):
        if metric == "KS":

            def cdf(x):
                hist_arr = np.array(hist)
                cum_hist_arr = hist_arr.cumsum()
                n_bins = len(hist_arr)
                bin_width = 2 / n_bins
                x_bin = np.floor_divide(x + 1, bin_width).astype(int)
                cdf_vals = cum_hist_arr[x_bin]
                return cdf_vals

            stat, pval = ks_1samp(self.members, cdf)
            return pval
        else:
            raise ValueError(
                "This metric is currently not available. Use 'KS' instead."
            )
