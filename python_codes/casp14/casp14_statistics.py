import pickle
from typing import Dict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PREDICTOR_KEY = "TS427"
WORKDIR = Path(f"../../workdir/casp14/{PREDICTOR_KEY}_results/")


class Statistics:

    def __init__(self):

        self.line_names = list()
        self.median_scores = list()
        self.correlation_mxs = list()
        self.score_names = None

    def update_from_dict(self, instance_name: str, per_resi_scores: Dict[str, Dict[str, float]]):

        # Establishing all the score names.
        if self.score_names is None:
            self.score_names = np.array(list(per_resi_scores.keys()))

        # Add the name of the instance (structure id + index) to the list of names.
        self.line_names.append(instance_name)

        # Creating a unified residue id set.
        # Filter out residue ids that belong to None scores
        #  (of any type).
        resi_ids = None
        for score_name in self.score_names:

            valid_keys = {
                key
                for key, value in per_resi_scores[score_name].items()
                if value is not None
            }

            if resi_ids is None:
                resi_ids = valid_keys
            else:
                resi_ids.intersection_update(valid_keys)

        resi_ids = list(resi_ids)

        # Collecting all the scores into a single ndarray.
        all_scores = np.array([
            [
                per_resi_scores[score_name][resi_id]
                for resi_id in resi_ids
            ]
            for score_name in self.score_names
        ])

        # Calculating the per-residue median value for every score type.
        self.median_scores.append(
            np.median(all_scores, axis=1)
        )

        self.correlation_mxs.append(
            spearmanr(all_scores, axis=1).statistic
        )

    def calculate_gaps(self):

        # Organize the same-structure statistical descriptors into groups.
        all_groups = dict()
        iterator = zip(self.line_names, self.median_scores, self.correlation_mxs)
        for name, median_score, correlation_mx in iterator:

            structure_name, structure_number = name.split("_")

            current_group = all_groups.get(
                structure_name,
                {"numbers": list(), "median": list(), "corr_mx": list()}
            )

            current_group["numbers"].append(structure_number)
            current_group["median"].append(median_score)
            current_group["corr_mx"].append(correlation_mx)

            all_groups[structure_name] = current_group

        # Calculate gaps within the groups.
        for structure_name in all_groups:

            group_numbers = np.array(all_groups[structure_name]["numbers"])

            # First, deal with the per-residue median scores.
            group_medians = np.array(all_groups[structure_name]["median"])

            group_max_median_idxs = np.argmax(group_medians, axis=0)
            group_min_median_idxs = np.argmin(group_medians, axis=0)

            arange = np.arange(len(self.score_names))
            group_max_median_scores = group_medians[group_max_median_idxs, arange]
            group_min_median_scores = group_medians[group_min_median_idxs, arange]
            group_median_score_gaps = group_max_median_scores - group_min_median_scores

            group_max_median_numbers = group_numbers[group_max_median_idxs]
            group_min_median_numbers = group_numbers[group_min_median_idxs]

            group_median_gap_data = [
                (gap, num1, num2)
                for gap, num1, num2 in
                zip(group_median_score_gaps, group_max_median_numbers, group_min_median_numbers)
            ]

            # Next, deal with the correlation matrices
            group_corr_mxs = np.array(all_groups[structure_name]["corr_mx"])

            for idx1, sn1 in enumerate(self.score_names):
                for idx2, sn2 in enumerate(self.score_names[idx1 + 1:]):
                    idx2 += idx1 + 1
                    group_max_corr_idx = np.argmax(group_corr_mxs[:, idx1, idx2])
                    group_min_corr_idx = np.argmin(group_corr_mxs[:, idx1, idx2])
                    # TODO: continue here!

    def summary(self):

        meadian_summary = {
            "min": np.min(self.median_scores, axis=0),
            "max": np.max(self.median_scores, axis=0),
            "mean": np.mean(self.median_scores, axis=0),
            "median": np.median(self.median_scores, axis=0),
            "StDev": np.std(self.median_scores, axis=0),
        }

        correlation_mx_summary = {
            "min": np.min(self.correlation_mxs, axis=0),
            "max": np.max(self.correlation_mxs, axis=0),
            "mean": np.mean(self.correlation_mxs, axis=0),
            "median": np.median(self.correlation_mxs, axis=0),
            "StDev": np.std(self.correlation_mxs, axis=0),
        }

        return {
            "median_summary": meadian_summary,
            "correlation_mx_summary": correlation_mx_summary
        }


def main():

    # Read the score collection from the pickled file
    with open(WORKDIR / f"{PREDICTOR_KEY}_ost_results_extended.pickle", "rb") as f:
        score_collection = pickle.load(f)

    # Initializing the statistics object.
    stats = Statistics()

    # Loop through all LoCoHD-score datasets.
    for structure_name, structure_options in score_collection.items():
        for structure_number, current_scores in structure_options.items():

            stats.update_from_dict(
                f"{structure_name}{PREDICTOR_KEY}_{structure_number}",
                current_scores["per_residue"]
            )

    stats.calculate_gaps()
    summary = stats.summary()


if __name__ == "__main__":
    main()
