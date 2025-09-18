import os.path
import pickle
import numpy as np
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt

from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import beta as beta_dist
from scipy.stats import kstest

# Residue hydrophobicities:
# https://en.wikipedia.org/wiki/Hydrophobicity_scales#/media/File:Hydrophobicity_scales2.gif
# Residue sizes:
# https://www.imgt.org/IMGTeducation/Aide-memoire/_UK/aminoacids/IMGTclasses.html
# Secondary structure propensities:
# https://bmcstructbiol.biomedcentral.com/articles/10.1186/1472-6807-12-18/tables/2

PROPERTY_NAMES = [
    "full", "resname", "charge", "aromaticity",
    "hydrophobicity", "size", "helix_propensity",
    "sheet_propensity"
]
RESI_PROPERTIES = {
    "GLY": ["-", "GLY", "neutral", "non-aromatic", "hydroneutral", "small", "low", "low"],
    "ALA": ["-", "ALA", "neutral", "non-aromatic", "hydroneutral", "small", "high", "low"],
    "VAL": ["-", "VAL", "neutral", "non-aromatic", "hydroneutral", "medium-size", "intermediate", "high"],
    "ILE": ["-", "ILE", "neutral", "non-aromatic", "hydrophobic", "large", "intermediate", "high"],
    "LEU": ["-", "LEU", "neutral", "non-aromatic", "hydrophobic", "large", "high", "intermediate"],
    "PHE": ["-", "PHE", "neutral", "aromatic", "hydrophobic", "large", "intermediate", "high"],
    "SER": ["-", "SER", "neutral", "non-aromatic", "hydroneutral", "small", "low", "intermediate"],
    "THR": ["-", "THR", "neutral", "non-aromatic", "hydroneutral", "small", "low", "intermediate"],
    "TYR": ["-", "TYR", "neutral", "aromatic", "hydrophobic", "large", "intermediate", "high"],
    "ASP": ["-", "ASP", "negative", "non-aromatic", "hydrophilic", "small", "low", "low"],
    "GLU": ["-", "GLU", "negative", "non-aromatic", "hydrophilic", "medium-size", "high", "low"],
    "ASN": ["-", "ASN", "neutral", "non-aromatic", "hydrophilic", "small", "low", "low"],
    "GLN": ["-", "GLN", "neutral", "non-aromatic", "hydrophilic", "medium-size", "high", "low"],
    "CYS": ["-", "CYS", "neutral", "non-aromatic", "hydrophobic", "small", "low", "high"],
    "MET": ["-", "MET", "neutral", "non-aromatic", "hydrophobic", "large", "high", "intermediate"],
    "TRP": ["-", "TRP", "neutral", "aromatic", "hydrophobic", "large", "intermediate", "high"],
    "HIS": ["-", "HIS", "neutral", "aromatic", "hydroneutral", "medium-size", "intermediate", "intermediate"],
    "ARG": ["-", "ARG", "positive", "non-aromatic", "hydrophilic", "large", "high", "intermediate"],
    "LYS": ["-", "LYS", "positive", "non-aromatic", "hydrophilic", "large", "intermediate", "intermediate"],
    "PRO": ["-", "PRO", "neutral", "non-aromatic", "hydrophilic", "small", "low", "low"]
}
DATA_SOURCE_DIR = Path("../../workdir/pisces/Examples_book")
DATA_SOURCE_NAME = "250917_KS_CGCent_K-3-10-3-6"
FIT_BETA = True
MM_TO_INCH = 0.0393701

def arg_median(x):
    return np.argpartition(x, len(x) // 2)[len(x) // 2]


def generate_statistics(data: List[Tuple[str, str, float]]) -> List[Dict[Tuple[str, str], Dict[str, Any]]]:
    """
    Measures different statistical descriptors for different property type pairs within
    a property category.
    :param data:
    :return:
    """

    # statistics is a list of dictionaries. Each element corresponds to a certain
    #  property category from PROPERTY_NAMES (e.g. hydrophobicity). The keys of the
    #  dictionaries are property pairs for the given category (e.g. hydrophilic-hydrophobic,
    #  hydrophobic-hydroneutral, etc...). The values are lists, that keep track of the observations.
    statistics = [dict() for _ in PROPERTY_NAMES]

    for progress, element in enumerate(data):

        if progress % 1000 == 0:
            print(f"\rStatistics gathering progress: {progress / len(data):.1%}", end="")

        resi_name1 = element[0][-3:]
        resi_name2 = element[1][-3:]

        # These are the property lists of the two amino acids.
        props1 = RESI_PROPERTIES[resi_name1]
        props2 = RESI_PROPERTIES[resi_name2]

        for idx, (prop1, prop2) in enumerate(zip(props1, props2)):

            key = (prop1, prop2) if prop1 > prop2 else (prop2, prop1)  # make it unambiguous

            if key not in statistics[idx]:
                statistics[idx][key] = list()

            statistics[idx][key].append(element)

    # Convert from AdvancedWelfordStatistics to exact statistical descriptors.
    for category_stat in statistics:

        for cat_key in category_stat:

            scores = np.array([x[2] for x in category_stat[cat_key]])

            cat_len = len(scores)
            cat_mean = np.mean(scores)
            cat_std = np.std(scores, mean=cat_mean)
            cat_arg_median = arg_median(scores)
            cat_median = category_stat[cat_key][cat_arg_median]
            cat_arg_min = np.argmin(scores)
            cat_min = category_stat[cat_key][cat_arg_min]
            cat_arg_max = np.argmax(scores)
            cat_max = category_stat[cat_key][cat_arg_max]

            category_stat[cat_key] = {
                "number of samples": cat_len,
                "mean": cat_mean,
                "median": cat_median,
                "standard deviation": cat_std,
                "minimum": cat_min,
                "maximum": cat_max
            }

    print()
    return statistics


def stat_to_tsvs(statistics: List[Dict[Tuple[str, str], Dict[str, Any]]]) -> List[str]:
    """
    Creates several tsv file texts based on the statistics provided from the
    generate_statistics function.
    :param statistics:
    :return:
    """

    # Create the tsv formatted string for every property category.
    output = ["" for _ in PROPERTY_NAMES]
    for prop_name_idx, category_stat in enumerate(statistics):

        # Convert the category statistics dictionaries to lists of items and
        #  sort them based on the mean values.
        category_stat_items: List[Tuple[Tuple[str, str], Dict[str, Any]]]
        category_stat_items = list(category_stat.items())
        category_stat_items.sort(key=lambda x: x[1]["mean"])

        # Create tsv header
        output[prop_name_idx] += "type 1\ttype 2\t"
        output[prop_name_idx] += "number of samples\t"
        output[prop_name_idx] += "mean\t"
        output[prop_name_idx] += "median id 1\tmedian id 2\tmedian\t"
        output[prop_name_idx] += "standard deviation\t"
        output[prop_name_idx] += "confidence interval\t"
        output[prop_name_idx] += "minimum id 1\tminimum id 2\tminimum\t"
        output[prop_name_idx] += "maximum id 1\tmaximum id 2\tmaximum\n"

        # Fill the tsv with data
        for (prop_type1, prop_type2), prop_type_stat in category_stat_items:

            conf_interval = stats.t.ppf(0.95, prop_type_stat["number of samples"] - 1)
            conf_interval *= prop_type_stat["standard deviation"]
            conf_interval /= math.sqrt(prop_type_stat["number of samples"])

            output[prop_name_idx] += f"{prop_type1}\t{prop_type2}\t"
            output[prop_name_idx] += f"{prop_type_stat['number of samples']}\t"
            output[prop_name_idx] += f"{prop_type_stat['mean']}\t"
            output[prop_name_idx] += f"{prop_type_stat['median'][0]}\t"
            output[prop_name_idx] += f"{prop_type_stat['median'][1]}\t"
            output[prop_name_idx] += f"{prop_type_stat['median'][2]}\t"
            output[prop_name_idx] += f"{prop_type_stat['standard deviation']}\t"
            output[prop_name_idx] += f"{conf_interval}\t"
            output[prop_name_idx] += f"{prop_type_stat['minimum'][0]}\t"
            output[prop_name_idx] += f"{prop_type_stat['minimum'][1]}\t"
            output[prop_name_idx] += f"{prop_type_stat['minimum'][2]}\t"
            output[prop_name_idx] += f"{prop_type_stat['maximum'][0]}\t"
            output[prop_name_idx] += f"{prop_type_stat['maximum'][1]}\t"
            output[prop_name_idx] += f"{prop_type_stat['maximum'][2]}\n"

    return output


def fit_beta_to_samples(lchd_values: List[float], analysis_dir_path: Path):

    def beta_cdf(x, *params) -> float:
        return beta_dist.cdf(x, a=params[0], b=params[1])

    x_values = np.sort(lchd_values)
    y_values = np.array(list(range(len(x_values)))) / len(x_values)

    start_params = [2., 2.]
    bound_params = ([0., 0.], [1E5, 1E5])
    fit_results = curve_fit(beta_cdf, x_values, y_values, p0=start_params, bounds=bound_params)[0]

    out_str = ""
    out_str += f"Beta distribution parameters:\n"
    out_str += f"\talpha = {fit_results[0]:.5f}\n"
    out_str += f"\tbeta = {fit_results[1]:.5f}\n"

    kstest_result = kstest(lchd_values, lambda x: beta_dist.cdf(x, *fit_results))
    out_str += f"Kolmogorov-Smirnov test result:\n"
    out_str += f"\tstatistics = {kstest_result[0]}\n"
    out_str += f"\tp-value = {kstest_result[1]}\n"

    with open(analysis_dir_path / "fitting_params.txt", "w") as f:
        f.write(out_str)

    print(out_str)

    # Plotting the full distribution histogram, along with the fitted beta distribution PDF.
    print("Beta-distribution fitted and saved!")

    return fit_results


def plot_results(lchd_scores: List[float], analysis_dir_path: Path):

    plt.rcParams["font.size"] = 7
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["figure.subplot.left"] = 0.2
    plt.rcParams["figure.subplot.right"] = 0.8
    plt.rcParams["figure.subplot.bottom"] = 0.15

    fig, ax = plt.subplots()

    # Histogram, boxplot, median text
    ax.set_xlabel("LoCoSD score")
    ax.set_ylabel("Count", color="blue")

    hist_lchd_y, hist_lchd_x = np.histogram(lchd_scores, bins=100, density=False)
    hist_lchd_x = (hist_lchd_x[1:] + hist_lchd_x[:-1]) / 2
    hist_max_y = np.max(hist_lchd_y)
    ax.plot(
        hist_lchd_x, hist_lchd_y,
        label="experimental distribution",
        color="blue"
    )

    box_dict = ax.boxplot(
        lchd_scores,
        positions=[hist_max_y * 1.1, ],
        widths=[hist_max_y * 0.05, ],
        vert=False, showfliers=False, manage_ticks=False
    )
    box_dict["medians"][0].set_color("blue")
    median_line_top = box_dict["medians"][0].get_xydata()[1, :]

    ax.text(
        median_line_top[0], median_line_top[1] + hist_max_y * 0.02,
        f"median:\n{median_line_top[0]:.5f}",
        ha="center", va="bottom",
                            )

    # Plot CDF too
    ax_cdf = ax.twinx()
    ax_cdf.set_ylabel("Cumulative Density", color="red")
    hist_lchd_y_cdf = np.cumsum(hist_lchd_y)
    hist_lchd_y_cdf = hist_lchd_y_cdf / hist_lchd_y_cdf[-1]
    ax_cdf.plot(
        hist_lchd_x,  hist_lchd_y_cdf,
        label="experimental cumulative distribution",
        color="red"
    )
    ax_cdf.set_yticks(np.arange(0, 1.5 + 0.01, 0.25))
    ax_cdf.set_ylim(0, 1.5)

    # Beta distribution fitting, if needed
    if FIT_BETA:

        print("tsv tables saved! Fitting beta-distribution...")
        beta_params = fit_beta_to_samples(lchd_scores, analysis_dir_path)
        beta_plot_x = np.arange(0, ax.get_xlim()[1] + 0.01, 0.01)
        beta_plot_y = beta_dist.pdf(beta_plot_x, *beta_params)
        ax.plot(
            beta_plot_x, beta_plot_y * hist_max_y / np.max(beta_plot_y),
            alpha=0.5, color="grey"
        )

    plot_y_ticks = np.arange(0, hist_max_y * 1.5, 5000)
    ax.set_yticks(plot_y_ticks)
    ax.set_ylim(0, ax.get_ylim()[1])

    fig.set_size_inches(88 * MM_TO_INCH, 88 * MM_TO_INCH)
    fig.savefig(analysis_dir_path / "distribution.svg", dpi=300)


def main():

    print(f"Opening directory: \"{DATA_SOURCE_NAME}\" at \"{DATA_SOURCE_DIR}\"")

    with open(DATA_SOURCE_DIR / DATA_SOURCE_NAME / "locohd_data.pisces", "rb") as f:
        data: List[Tuple[str, str, float]] = pickle.load(f)

    print(f"Number of samples: {len(data)}. Starting to generate statistics...")

    # Generating statistics for different property type pairs.
    statistics = generate_statistics(data)

    print("Statistics generated! Creating tsvs...")
    tsvs = stat_to_tsvs(statistics)

    # Saving the statistics.
    print("tsvs created! Saving tsv tables...")
    analysis_dir_path = DATA_SOURCE_DIR / DATA_SOURCE_NAME / "analysis"
    if not os.path.exists(analysis_dir_path):
        os.mkdir(analysis_dir_path)

    for prop_name, tsv_data in zip(PROPERTY_NAMES, tsvs):
        with open(analysis_dir_path / f"{prop_name}_statistics.tsv", "w") as f:
            f.write(tsv_data)

    # Plotting
    lchd_scores = list(map(lambda x: x[2], data))

    print("Starting to plot...")
    plot_results(lchd_scores, analysis_dir_path)


if __name__ == "__main__":
    main()
