import pickle
import numpy as np
import math
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import beta as beta_dist
from scipy.stats import kstest


def source_to_name(source: Path) -> str:

    with open(source / "params.json", "r") as f:
        source_data = json.load(f)

    primitive_typing = Path(source_data["assigner_config_path"])
    primitive_typing = primitive_typing.name.replace(".config.json", "").replace("_", "-")

    weight_function = source_data["weight_function"]
    weight_function = weight_function[0] + "-" + "-".join(map(str, weight_function[1]))

    contacts = "only-hetero-contacts" if source_data["only_hetero_contacts"] else "all-contacts"

    return "_".join([primitive_typing, weight_function, contacts])


def fit_beta_to_samples(lchd_values: List[float]):

    def beta_cdf(x, *params) -> float:
        return beta_dist.cdf(x, a=params[0], b=params[1])

    x_values = np.sort(lchd_values)
    y_values = np.array(list(range(len(x_values)))) / len(x_values)

    start_params = [2., 2.]
    bound_params = ([0., 0.], [1E5, 1E5])
    fit_results = curve_fit(beta_cdf, x_values, y_values, p0=start_params, bounds=bound_params)

    return fit_results[0]


def main():

    out_str = ""

    workdir = Path("workdir/pisces")
    data_source = "run_2023-01-10-11-52-33"

    run_name = source_to_name(workdir / data_source)

    out_str += f"Opening directory: \"{data_source}\" at \"{workdir}\"\n"

    with open(workdir / data_source / "locohd_data.pisces", "rb") as f:
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

    min_idx = np.argmin(lchd_scores)
    max_idx = np.argmax(lchd_scores)

    out_str += f"Global statistics:\n"
    out_str += f"\tmean: {np.mean(lchd_scores):.5%}\n"
    out_str += f"\tmedian: {np.median(lchd_scores):.5%}\n"
    out_str += f"\tstd: {np.std(lchd_scores):.5%}\n"
    out_str += f"\tmin: {lchd_scores[min_idx]:.5%} (at {data[min_idx][0]} and {data[min_idx][1]})\n"
    out_str += f"\tmax: {lchd_scores[max_idx]:.5%} (at {data[max_idx][0]} and {data[max_idx][1]})\n"

    beta_params = fit_beta_to_samples(lchd_scores)

    out_str += f"Beta distribution parameters:\n"
    out_str += f"\talpha = {beta_params[0]:.5f}\n"
    out_str += f"\tbeta = {beta_params[1]:.5f}\n"

    kstest_result = kstest(lchd_scores, lambda x: beta_dist.cdf(x, *beta_params))
    out_str += f"Kolmogorov-Smirnov test result:\n"
    out_str += f"\tstatistics = {kstest_result[0]}\n"
    out_str += f"\tp-value = {kstest_result[1]}\n"

    with open(workdir / data_source / (run_name + ".analysis"), "w") as f:
        f.write(out_str)

    print(out_str)

    fig, ax = plt.subplots()

    ax.hist(lchd_scores, bins=100, density=True, label="experimental distribution")
    plot_range = np.arange(0, 1 + 0.01, 0.01)
    ax.plot(plot_range, beta_dist.pdf(plot_range, *beta_params),
            alpha=0.7,
            label="fitted $\\beta$-distribution")
    ax.legend(loc="upper right", shadow=True)
    ax.set_xlabel("LoCoHD score")
    ax.set_ylabel("Density")
    x_ticks = np.arange(0, 1 + 0.2, 0.2)
    ax.set_xticks(x_ticks, [f"{x:.0%}" for x in x_ticks])
    fig.savefig(workdir / data_source / (run_name + ".histogram.png"), dpi=300)


if __name__ == "__main__":
    main()
