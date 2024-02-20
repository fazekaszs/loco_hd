import pickle
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

from config import PREDICTOR_KEY, EXTRACTOR_OUTPUT_DIR
MM_TO_INCH = 0.0393701


class Histogram:

    def __init__(self, hist_bins, hist_range):

        self.hist_bins, self.hist_range = hist_bins, hist_range

        self.counts, self.x, self.y = np.histogram2d(
            [], [],
            bins=self.hist_bins,
            range=self.hist_range
        )

    def update(self, score1, score2):

        new_counts, _, _ = np.histogram2d(
            score1, score2,
            bins=self.hist_bins,
            range=self.hist_range
        )
        self.counts += new_counts

    def plot(
        self,
        plot_title: str,
        plot_save_name: str,
        score1_name: str,
        score2_name: str,
    ):

        fig, ax = plt.subplots()
        fig.suptitle(plot_title)
        fig.set_size_inches(88 * MM_TO_INCH, 88 * MM_TO_INCH)

        ax.imshow(self.counts[:, ::-1].T, cmap="hot")

        ticks = list(range(0, self.hist_bins, 10))
        ax.set_xticks(ticks, labels=[f"{self.x[idx]:.0%}" for idx in ticks])
        ax.set_yticks(ticks, labels=[f"{self.y[-idx-1]:.0%}" for idx in ticks])
        ax.set_xlabel(f"{score1_name} score")
        ax.set_ylabel(f"{score2_name} score")

        fig.savefig(EXTRACTOR_OUTPUT_DIR / "plots" / f"{plot_save_name}.svg", dpi=300)
        plt.close(fig)


def get_plot_alpha(
        score1, score2,
        threshold_dist: float = 0.01
):

    all_scores = np.array([score1, score2]).T
    min_scores = np.min(all_scores, axis=0)
    max_scores = np.max(all_scores, axis=0)
    all_scores = (all_scores - min_scores) / (max_scores - min_scores)

    dmx = all_scores[:, np.newaxis, :] - all_scores[np.newaxis, :, :]
    dmx = np.sqrt(np.sum(dmx ** 2, axis=2))

    alphas = 1 / (1 + (dmx / threshold_dist) ** 4)
    alphas = np.sum(alphas, axis=1)
    alphas = 1 / alphas

    return alphas


def create_plot(
        plot_title: str,
        plot_save_name: str,
        score1_name: str,
        score1_values: np.ndarray,
        score1_range: Tuple[float, ...],
        score2_name: str,
        score2_values: np.ndarray,
        score2_range: Tuple[float, ...],
):

    x_ticks = np.arange(*score1_range)
    y_ticks = np.arange(*score2_range)

    spr = spearmanr(score1_values, score2_values).correlation

    fig, ax = plt.subplots()
    fig.suptitle(plot_title)
    fig.set_size_inches(88 * MM_TO_INCH, 88 * MM_TO_INCH)

    ax.scatter(
        score1_values, score2_values,
        alpha=get_plot_alpha(score1_values, score2_values),
        edgecolors="none", c="red", marker=".", s=10
    )

    ax.set_xticks(x_ticks, labels=[f"{tick:.0%}" for tick in x_ticks])
    ax.set_yticks(y_ticks, labels=[f"{tick:.0%}" for tick in y_ticks])
    ax.set_xlim(score1_range[:2])
    ax.set_ylim(score2_range[:2])

    ax.set_xlabel(f"{score1_name} score")
    ax.set_ylabel(f"{score2_name} score")

    legend_labels = list()
    legend_labels.append(f"SpR = {spr:.4f}")
    legend_labels.append(f"mean {score1_name} = {np.mean(score1_values):.2%}")
    legend_labels.append(f"median {score1_name} = {np.median(score1_values):.2%}")
    legend_labels.append(f"mean {score2_name} = {np.mean(score2_values):.2%}")
    legend_labels.append(f"median {score2_name} = {np.median(score2_values):.2%}")

    legend_handles = [
        Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0),
    ] * len(legend_labels)

    ax.legend(
        legend_handles, legend_labels,
        loc="upper right", fontsize="small", fancybox=True,
        framealpha=0.7, handlelength=0, handletextpad=0
    )

    fig.savefig(EXTRACTOR_OUTPUT_DIR / "plots" / f"{plot_save_name}.svg", dpi=300)

    plt.close(fig)


def main():

    plt.rcParams["font.size"] = 7
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["figure.subplot.left"] = 0.2
    plt.rcParams["figure.subplot.bottom"] = 0.15

    # Read the score collection from the pickled file
    with open(EXTRACTOR_OUTPUT_DIR / f"ost_results_extended.pickle", "rb") as f:
        score_collection = pickle.load(f)

    # Create the 2D histograms for the specific score types.
    # In all of these, the first axis is along the LoCoHD scores, while
    #  the second axis is along the corresponding score.
    hists = {
        "cad_score": Histogram(100, [[0., 0.5], [0., 1.]]),
        "lddt": Histogram(100, [[0., 0.5], [0., 1.]]),
    }

    # The dictionary assigning a "normal", "plotting" score name to the "raw" score names.
    plot_score_name = {
        "cad_score": "CAD", "lddt": "lDDT"
    }

    # Loop through all LoCoHD-score datasets.
    for structure_name, structure_options in score_collection.items():
        for structure_number, current_scores in structure_options.items():

            # For each specific score...
            for score_name, per_resi_scores in current_scores["per_residue"].items():

                if score_name == "LoCoHD":
                    continue

                locohd_scores = current_scores["per_residue"]["LoCoHD"]

                print(f"\rI am at: {structure_name} {structure_number} {score_name}", end="")

                points = np.array([
                    (locohd_scores[resi_key], per_resi_scores[resi_key])
                    for resi_key in per_resi_scores
                    if per_resi_scores[resi_key] is not None
                ])

                create_plot(
                    f"predictor: {PREDICTOR_KEY}, structure: {structure_name}, index: {structure_number}",
                    f"{score_name}_{structure_name}{PREDICTOR_KEY}_{structure_number}",
                    "LoCoHD",
                    points[:, 0],
                    (0., 0.5, 0.05),
                    plot_score_name[score_name],
                    points[:, 1],
                    (0., 1., 0.1)
                )

                hists[score_name].update(points[:, 0], points[:, 1])

    # Plot each score's histogram.
    for score_name, hist in hists.items():
        hist.plot(
            f"Distribution of scores for\ncontestant {PREDICTOR_KEY}",
            f"{PREDICTOR_KEY}_{plot_score_name[score_name]}_full_hist",
            "LoCoHD",
            plot_score_name[score_name]
        )


if __name__ == "__main__":
    main()
