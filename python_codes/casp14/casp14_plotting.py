import pickle
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

PREDICTOR_KEY = "TS427"
WORKDIR = Path(f"../../workdir/casp14/{PREDICTOR_KEY}_results/")


def get_plot_alpha(score1, score2):

    area_per_tick = np.max(score1) - np.min(score1)
    area_per_tick *= np.max(score2) - np.min(score2)
    area_per_tick /= len(score2)
    area_per_tick *= 1E3

    return area_per_tick if area_per_tick < 1. else 1.


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

    ax.scatter(
        score1_values, score2_values,
        alpha=get_plot_alpha(score1_values, score2_values),
        edgecolors="none", c="red", marker="."
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

    fig.savefig(WORKDIR / "plots" / f"{plot_save_name}.svg")

    plt.close(fig)


def main():

    with open(WORKDIR / f"{PREDICTOR_KEY}_ost_results_extended.pickle", "rb") as f:
        score_collection = pickle.load(f)

    for structure_id, structure_options in score_collection.items():
        for predicted_idx, current_scores in structure_options.items():

            current_scores = current_scores["per_residue"]

            for score_name, per_resi_scores in current_scores.items():

                if score_name == "LoCoHD":
                    continue

                print(f"{structure_id} {predicted_idx} {score_name}")

                points = np.array([
                    (current_scores["LoCoHD"][resi_key], per_resi_scores[resi_key])
                    for resi_key in per_resi_scores
                    if per_resi_scores[resi_key] is not None
                ])

                plot_score_name = {
                    "cad_score": "CAD", "lddt": "lDDT"
                }[score_name]

                create_plot(
                    f"predictor: {PREDICTOR_KEY}, structure: {structure_id}, index: {predicted_idx}",
                    f"{score_name}_{structure_id}{PREDICTOR_KEY}_{predicted_idx}",
                    "LoCoHD",
                    points[:, 0],
                    (0., 0.5, 0.05),
                    plot_score_name,
                    points[:, 1],
                    (0., 1.0, 0.1)
                )


if __name__ == "__main__":
    main()
