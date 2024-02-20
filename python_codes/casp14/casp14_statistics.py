import pickle
from typing import Dict

import numpy as np
from scipy.stats import spearmanr

from config import PREDICTOR_KEY, EXTRACTOR_OUTPUT_DIR


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

        # Calculating the per-residue correlation matrix between different score types.
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
                {"numbers": list(), "median": list(), "corrmx": list()}
            )

            current_group["numbers"].append(structure_number)
            current_group["median"].append(median_score)
            current_group["corrmx"].append(correlation_mx)

            all_groups[structure_name] = current_group

        # Calculate gaps within the groups.
        for structure_name in all_groups:

            # Convert lists of strs and lists of ndarrays into ndarrays.
            group_numbers = np.array(all_groups[structure_name]["numbers"])  # N strings
            group_medians = np.array(all_groups[structure_name]["median"])  # (N, M) for the M scores
            group_corr_mxs = np.array(all_groups[structure_name]["corrmx"])  # (N, M, M), corr. between scores

            # First, deal with the per-residue median scores.
            group_max_median_idxs = np.argmax(group_medians, axis=0)
            group_min_median_idxs = np.argmin(group_medians, axis=0)

            arange = np.arange(len(self.score_names))
            group_max_median_scores = group_medians[group_max_median_idxs, arange]
            group_min_median_scores = group_medians[group_min_median_idxs, arange]
            group_median_score_gaps = group_max_median_scores - group_min_median_scores

            group_max_median_numbers = group_numbers[group_max_median_idxs]
            group_min_median_numbers = group_numbers[group_min_median_idxs]

            # Collect the largest score median gaps and the corresponding structure numbers.
            group_median_gap_data = {
                sn: (num1, num2, gap)
                for sn, num1, num2, gap in
                zip(
                    self.score_names,
                    group_max_median_numbers,
                    group_min_median_numbers,
                    group_median_score_gaps,
                )
            }
            all_groups[structure_name]["median_gaps"] = group_median_gap_data

            # Next, deal with the per-residue score correlation matrices.
            triu_ax1, triu_ax2 = np.triu_indices(group_corr_mxs.shape[1], k=1)

            group_max_corr_idx = np.argmax(group_corr_mxs, axis=0)[triu_ax1, triu_ax2]
            group_min_corr_idx = np.argmin(group_corr_mxs, axis=0)[triu_ax1, triu_ax2]

            group_max_corr_values = group_corr_mxs[group_max_corr_idx, triu_ax1, triu_ax2]
            group_min_corr_values = group_corr_mxs[group_min_corr_idx, triu_ax1, triu_ax2]
            group_corr_value_gaps = group_max_corr_values - group_min_corr_values

            group_max_corr_numbers = group_numbers[group_max_corr_idx]
            group_min_corr_numbers = group_numbers[group_min_corr_idx]

            # Collect the largest score correlation gaps and the corresponding structure numbers.
            group_corr_gap_data = {
                frozenset({sn1, sn2}): (num1, num2, gap)
                for sn1, sn2, num1, num2, gap in
                zip(
                    self.score_names[triu_ax1],
                    self.score_names[triu_ax2],
                    group_max_corr_numbers,
                    group_min_corr_numbers,
                    group_corr_value_gaps,
                )
            }
            all_groups[structure_name]["corrmx_gaps"] = group_corr_gap_data

            # Delete redundant data
            del all_groups[structure_name]["numbers"]
            del all_groups[structure_name]["median"]
            del all_groups[structure_name]["corrmx"]

        # Extract the maximum gap cases
        max_gaps = {
            "median_gaps": dict(), "corrmx_gaps": dict()
        }
        for structure_name in all_groups:

            # Again, deal with the score median gaps first.
            for score_name, gap_info in all_groups[structure_name]["median_gaps"].items():

                current_max = max_gaps["median_gaps"].get(
                    score_name,
                    ("", "", "", float("-inf"))
                )

                if gap_info[-1] > current_max[-1]:
                    max_gaps["median_gaps"][score_name] = (structure_name, *gap_info)

            for score_pair_key, gap_info in all_groups[structure_name]["corrmx_gaps"].items():

                current_max = max_gaps["corrmx_gaps"].get(
                    score_pair_key,
                    ("", "", "", float("-inf"))
                )

                if gap_info[-1] > current_max[-1]:
                    max_gaps["corrmx_gaps"][score_pair_key] = (structure_name, *gap_info)

        return max_gaps

    def summary(self):

        median_summary = {
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

        gaps = self.calculate_gaps()

        return {
            "score_names": self.score_names,
            "median_summary": median_summary,
            "correlation_mx_summary": correlation_mx_summary,
            **gaps
        }


def main():

    # Read the score collection from the pickled file
    with open(EXTRACTOR_OUTPUT_DIR / f"ost_results_extended.pickle", "rb") as f:
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

    summary = stats.summary()

    with open(EXTRACTOR_OUTPUT_DIR / f"statistics.pickle", "wb") as f:
        pickle.dump(summary, f)


if __name__ == "__main__":
    main()
