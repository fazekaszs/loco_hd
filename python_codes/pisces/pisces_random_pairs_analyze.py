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
DATA_SOURCE_DIR = Path("../../workdir/pisces")
DATA_SOURCE_NAME = "run_2023-02-08-12-50-23"
MM_TO_INCH = 0.0393701


class AdvancedWelfordStatistics:
    """
    Creates an object capable to collect statistics from a stream of data i.e.,
    updates and keeps track of the following descriptors in an online manner:
    - the number of samples seen
    - the mean of the samples (estimate)
    - the median of the samples (estimate)
    - the variance of the samples (estimate)
    - the minimum of the samples (exact)
    - the maximum of the samples (exact)

    It uses the Welford online algorithm for this. The samples are stored in a buffer
    vector, until the vector reaches its capacity. When it does, the descriptors are
    updated and the buffer vector is flushed. The higher the capacity, the more accurate
    are the estimated descriptors, but the slower the algorithm.

    Also, samples are added as tuples: the first two elements of the tuples are residue
    data in the form of "[PDB ID]/[chain ID]/[residue number]-[residue type]" strings. The
    third element is the measured LoCoHD score between the two residue environments.
    """

    def __init__(self, capacity: int = 1_000_000):

        self.capacity: int = capacity
        self.n_of_samples: int = 0
        self.buffer: List[Tuple[str, str, float]] = list()
        self.mean: float = 0
        self.full_var: float = 0
        self.min: Tuple[str, str, float] = ("", "", float("inf"))
        self.max: Tuple[str, str, float] = ("", "", float("-inf"))

        # This only estimates the median!
        self.medians: List[Tuple[str, str, float]] = list()

    def collapse(self):
        """
        Updates the statistical descriptors based on the data inside the buffer
        vector and then flushes the buffer.
        """

        # Each element is an (id1: str, id2: str, locohd: float) tuple in 'self.buffer'.
        buffer_array = np.array([element[2] for element in self.buffer])

        # Update the number of samples.
        self.n_of_samples += len(buffer_array)

        # Update the mean with the online algorithm:
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online
        # This is a modification of the algorithm described above.
        # Here, we add 'len(self.buffer)' number of samples each time.
        old_mean = self.mean
        self.mean = old_mean + (np.sum(buffer_array) - len(self.buffer) * old_mean) / self.n_of_samples

        # Update the squared deviation from the mean.
        self.full_var += np.sum((buffer_array - self.mean) * (buffer_array - old_mean))

        # Update the min value.
        current_min_idx = np.argmin(buffer_array)
        current_min = buffer_array[current_min_idx]
        if current_min < self.min[2]:
            self.min = self.buffer[current_min_idx]

        # Update the max value.
        current_max_idx = np.argmax(buffer_array)
        current_max = buffer_array[current_max_idx]
        if current_max > self.max[2]:
            self.max = self.buffer[current_max_idx]

        # Update the medians.
        self.medians += self.buffer
        self.medians.sort(key=lambda x: x[2])
        lower_cut = max((0, (len(self.medians) - self.capacity) // 2))
        upper_cut = min((len(self.medians), (len(self.medians) + self.capacity) // 2))
        self.medians = self.medians[lower_cut:upper_cut]

        # Clean up the buffer.
        self.buffer = list()

    def update(self, value: Tuple[str, str, float]):
        """
        Adds a new element to the buffer vector and then calls collapse if the length
        of the buffer vector exceeds the capacity.
        :param value: A new sample from the LoCoHD distribution along with the LoCoHD source
         (residue environment identifiers).
        """

        self.buffer.append(value)

        if len(self.buffer) == self.capacity:
            self.collapse()

    def get_stat(self):

        self.collapse()

        std = math.sqrt(self.full_var / self.n_of_samples)
        median = self.medians[len(self.medians) // 2]

        out = {
            "number of samples": self.n_of_samples,
            "mean": self.mean,
            "median": median,
            "standard deviation": std,
            "minimum": self.min,
            "maximum": self.max
        }

        return out


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
    #  hydrophobic-hydroneutral, etc...). The values are AdvancedWelfordStatistics instances,
    #  that keep track of the statistical descriptors of the samples.
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
                statistics[idx][key] = AdvancedWelfordStatistics()

            statistics[idx][key].update(element)

    # Convert from AdvancedWelfordStatistics to exact statistical descriptors.
    for category_stat in statistics:

        for cat_key in category_stat:
            category_stat[cat_key] = category_stat[cat_key].get_stat()

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

    # Fitting a beta-distribution to the data
    print("tsv tables saved! Fitting beta-distribution...")
    lchd_scores = list(map(lambda x: x[2], data))

    beta_params = fit_beta_to_samples(lchd_scores)

    out_str = ""
    out_str += f"Beta distribution parameters:\n"
    out_str += f"\talpha = {beta_params[0]:.5f}\n"
    out_str += f"\tbeta = {beta_params[1]:.5f}\n"

    kstest_result = kstest(lchd_scores, lambda x: beta_dist.cdf(x, *beta_params))
    out_str += f"Kolmogorov-Smirnov test result:\n"
    out_str += f"\tstatistics = {kstest_result[0]}\n"
    out_str += f"\tp-value = {kstest_result[1]}\n"

    with open(analysis_dir_path / "fitting_params.txt", "w") as f:
        f.write(out_str)

    print(out_str)

    # Plotting the full distribution histogram, along with the fitted beta distribution PDF.
    print("Beta-distribution fitted and saved! Starting to plot...")

    plt.rcParams["font.size"] = 7
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["figure.subplot.left"] = 0.2
    plt.rcParams["figure.subplot.bottom"] = 0.15

    fig, ax = plt.subplots()

    ax.hist(
        lchd_scores,
        bins=100, density=True, label="experimental distribution"
    )

    plot_x = np.arange(0, 1 + 0.01, 0.01)
    fitted_beta_y = beta_dist.pdf(plot_x, *beta_params)
    fitted_beta_height = np.max(fitted_beta_y)
    plot_y_ticks = np.arange(0, fitted_beta_height * 2, fitted_beta_height * 2 / 10)

    ax.plot(
        plot_x, fitted_beta_y,
        alpha=0.7, label="fitted $\\beta$-distribution"
    )
    box_dict = ax.boxplot(
        lchd_scores,
        positions=[fitted_beta_height * 1.3, ],
        widths=[fitted_beta_height * 0.1, ],
        vert=False, showfliers=False, manage_ticks=False
    )

    box_dict["medians"][0].set_color("blue")
    median_line_top = box_dict["medians"][0].get_xydata()[1, :]
    ax.text(
        median_line_top[0], median_line_top[1],
        f"median: {median_line_top[0]:.2%}",
        ha="center", va="bottom",
    )

    ax.legend(loc="upper right")
    ax.set_xlabel("LoCoHD score")
    ax.set_ylabel("Density")

    x_ticks = np.arange(0, 1 + 0.2, 0.2)
    ax.set_xticks(x_ticks, [f"{x:.0%}" for x in x_ticks])
    ax.set_yticks(plot_y_ticks)

    fig.set_size_inches(88 * MM_TO_INCH, 88 * MM_TO_INCH)
    fig.savefig(analysis_dir_path / "distribution.svg", dpi=300)


if __name__ == "__main__":
    main()
