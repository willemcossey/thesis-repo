from scipy.stats import ks_2samp


class SampleCollection:
    def __init__(self, lst: list):
        # check if list not nested
        self.members = lst

    # Takes in two collections of samples and returns the probabilitys
    # that both collections stem from the same distribution.
    def compare(self, other, metric="KS"):
        if metric == "KS":
            stat, pval = ks_2samp(self.members, other)
            return pval
        else:
            raise ValueError(
                "This metric is currently not available. Use 'KS' instead."
            )
