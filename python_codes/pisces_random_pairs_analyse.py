import pickle
import numpy as np
import math
from scipy import stats
from pathlib import Path

import matplotlib.pyplot as plt
from fitter import Fitter
from scipy.stats import beta as beta_dist
from scipy.stats import kstest


def main():

    out_str = ""

    workdir = Path("workdir/pisces")
    data_source = "results_uniform-3-10_only-hetero-contacts.pickle"
    # data_source = "results_uniform-3-10_all-contacts.pickle"
    # data_source = "results_kumaraswamy-3-10-2-5_only-hetero-contacts.pickle"
    data_source = "results_kumaraswamy-3-10-2-5_only-hetero-contacts_coarse.pickle"

    out_str += f"Opening file: \"{data_source}\" at \"{workdir}\"\n"

    with open(workdir / data_source, "rb") as f:
        data = pickle.load(f)

    out_str += f"Number of samples: {len(data)}\n"

    resi_stat = dict()
    for element in data:

        key = [element[0][-3:], element[1][-3:]]
        key = sorted(key)
        key = tuple(key)

        if key in resi_stat:
            resi_stat[key].append(element[2])
        else:
            resi_stat[key] = [element[2], ]

    for key in resi_stat:

        sample_mean = np.mean(resi_stat[key])
        sample_std = np.std(resi_stat[key])
        sample_conf_int = stats.t.ppf(0.95, len(resi_stat[key]) - 2) * sample_std / math.sqrt(len(resi_stat[key]))

        resi_stat[key] = (sample_mean, sample_std, sample_conf_int)

    resi_pairs = list(resi_stat.keys())
    resi_pairs = sorted(resi_pairs, key=lambda x: resi_stat[x][0])

    out_str += f"Average random residue-residue LoCoHD:\n"
    for key in resi_pairs:
        out_str += f"\t{key[0]}-{key[1]}: {resi_stat[key][0]:.2%} +/- {resi_stat[key][2]:.2%}\n"

    del resi_stat

    lchd_scores = list(map(lambda x: x[2], data))

    fitt = Fitter(lchd_scores)
    fitt.distributions = ["beta", ]
    fitt.fit()
    beta_params = fitt.fitted_param["beta"]

    del fitt

    out_str += f"Beta distribution parameters:\n"
    out_str += f"\talpha = {beta_params[0]:.5f}\n"
    out_str += f"\tbeta = {beta_params[1]:.5f}\n"
    out_str += f"\tlocation = {beta_params[2]:.5f}\n"
    out_str += f"\tscale = {beta_params[3]:.5f}\n"

    kstest_result = kstest(lchd_scores, lambda x: beta_dist.cdf(x, *beta_params))
    out_str += f"Kolmogorov-Smirnov test result:\n"
    out_str += f"\tstatistics = {kstest_result[0]}\n"
    out_str += f"\tp-value = {kstest_result[1]}\n"

    with open(workdir / (data_source + ".analysis"), "w") as f:
        f.write(out_str)

    fig, ax = plt.subplots()

    ax.hist(lchd_scores, bins=100, density=True)
    plot_range = np.arange(0, 1 + 0.01, 0.01)
    ax.plot(plot_range, beta_dist.pdf(plot_range, *beta_params))
    fig.savefig(workdir / (data_source + ".histogram.png"), dpi=300)


if __name__ == "__main__":
    main()
