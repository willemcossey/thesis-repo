import numpy as np
import torch
from helper.Distribution import Distribution, Uniform

n_test_samples = 10000
x_test = torch.ones(n_test_samples, 2)
x_test[:, 0] = torch.tensor(Uniform(0, 12).sample(n_test_samples))
x_test[:, 1] = torch.tensor(Uniform(-1, 1).sample(n_test_samples))
n_buckets = 20
h = 2 / n_buckets
centers = [-1 + h / 2 + i * h for i in range(n_buckets)]


def get_test_samples(n_test_samples, n_buckets):
    x_test = torch.ones(n_test_samples, 2)
    x_test[:, 0] = torch.tensor(Uniform(0, 12).sample(n_test_samples))
    x_test[:, 1] = torch.tensor(Uniform(-1, 1).sample(n_test_samples))
    y_test = np.empty([n_test_samples, n_buckets])

    def inv_dist(w, m, lam):
        if abs(w) == 1:
            return 0
        else:

            res1 = np.log(1 + w) * (-2 + (m / (2 * lam)))
            res2 = np.log(1 - w) * (-2 - (m / (2 * lam)))
            res3 = -((1 - m * w) / (lam * (1 - w**2)))
            if res1 + res2 + res3 > 700:
                # print("overflow")
                # print(w,m,lam, res1 + res2 + res3)
                return np.nan
            elif torch.exp(res1 + res2 + res3) == 0:
                # print("underflow")
                # print(w, m, lam, res1 + res2 + res3)
                return np.nan
            else:
                return torch.exp(res1 + res2 + res3)

    def inv_dist_norm_hist(m, lam, n_buckets):
        h = 2 / n_buckets
        centers = [-1 + h / 2 + i * h for i in range(n_buckets)]
        y = np.empty(n_buckets)

        for j in range(len(centers)):
            y[j] = inv_dist(centers[j], m, lam)

        if np.any(np.isnan(y)):
            return np.empty(y.shape).fill(np.nan)
        else:
            sum = y.sum()
            if sum is None or sum == 0 or sum == np.nan:
                print(sum)
                return np.empty(y.shape).fill(np.nan)
            else:
                return y / sum

    for i in range(n_test_samples):
        # print(x_test[i,1], x_test[i,0],n_buckets)
        prop = inv_dist_norm_hist(x_test[i, 1], x_test[i, 0], n_buckets)
        # print(prop)
        if prop is None or np.any(np.isnan(prop)):
            new_x, new_y = get_test_samples(1, n_buckets)
            x_test[i, :] = new_x
            y_test[i, :] = new_y
        else:
            y_test[i, :] = prop

    y_test = torch.tensor(y_test)

    return x_test, y_test


x_t, y_t = get_test_samples(1000, 20)

print(torch.any(torch.isnan(y_t)))
