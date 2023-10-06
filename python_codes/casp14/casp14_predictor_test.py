import tarfile
import codecs
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

from Bio.PDB.Structure import Structure

from pathlib import Path
from typing import List, Dict, Tuple
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

from loco_hd import LoCoHD, PrimitiveAtom, WeightFunction, PrimitiveAssigner, PrimitiveAtomTemplate, TagPairingRule


# Set the necessary paths. The available predictor keys are the following:
# AF2: TS427, BAKER: TS473, BAKER-experimental: TS403, FEIG-R2: TS480, Zhang: TS129
LDDT_TARS_PATH = Path("../../data_sources/casp14/lDDTs")
PREDICTOR_KEY = "TS129"
WORKDIR = Path(f"../workdir/casp14/{PREDICTOR_KEY}_results")

ScoreStatType = Dict[str, Dict[str, Tuple[float, float, float]]]


def prat_to_pra(prat: PrimitiveAtomTemplate) -> PrimitiveAtom:

    resi_id = prat.atom_source.source_residue
    resname = prat.atom_source.source_residue_name
    source = f"{resi_id[2]}/{resi_id[3][1]}-{resname}"
    return PrimitiveAtom(prat.primitive_type, source, prat.coordinates)


def lddt_from_text(text: str):

    text = text.split("\n")
    header_text = "Chain	ResName	ResNum	Asses.	Q.Prob.	Score"
    start_idx = next(idx for idx in range(len(text)) if text[idx].startswith(header_text))
    text = text[start_idx + 1:]
    text = list(map(lambda line: line.split("\t"), text))
    text = {f"{line[0]}/{line[2]}-{line[1]}": float(line[5]) for line in text if len(line) != 1 and line[5] != "-"}
    return text


def read_lddt_values(structure_key: str) -> Dict[str, Dict[str, float]]:

    lddt_dict = dict()
    with tarfile.open(LDDT_TARS_PATH / f"{structure_key}.tgz") as tf:
        tf_members = tf.getmembers()
        tf_members = list(filter(lambda m: f"{structure_key}{PREDICTOR_KEY}_" in m.name, tf_members))
        for tf_member in tf_members:
            f = tf.extractfile(tf_member)
            content = codecs.getreader("utf-8")(f).read()
            content = lddt_from_text(content)
            member_key = tf_member.name.split("/")[2].replace(".lddt", "")
            lddt_dict[member_key] = content
            f.close()

    return lddt_dict


def get_statistics_str(stat_dict: ScoreStatType) -> str:

    def data_to_table_row(data) -> str:
        return f"{data[0]} | " + " | ".join(map(lambda x: f"{x:.7f}", data[1:])) + " |\n"

    stat_list = list()
    for structure_key in stat_dict:
        for pred_id in stat_dict[structure_key]:
            stat_list.append((f"{structure_key}_{pred_id}", *stat_dict[structure_key][pred_id]))

    out_str = ""

    # SpR statistics
    spr_argmedian = np.argsort([x[1] for x in stat_list])[len(stat_list) // 2]
    spr_argmin = np.argmin([x[1] for x in stat_list])
    spr_argmax = np.argmax([x[1] for x in stat_list])

    out_str += "## Statistics for the Spearman's rank-Correlation Coefficient\n"
    out_str += f"- mean: {np.mean([x[1] for x in stat_list]):.5f}\n"
    out_str += f"- std: {np.std([x[1] for x in stat_list]):.5f}\n\n"

    out_str += "| descriptor | structure | SpR | median LoCoHD | median lDDT |\n"
    out_str += "| ---------: | :-------: | :-: | :-----------: | :---------: |\n"
    out_str += f"| median | {data_to_table_row(stat_list[spr_argmedian])}"
    out_str += f"| min | {data_to_table_row(stat_list[spr_argmin])}"
    out_str += f"| max | {data_to_table_row(stat_list[spr_argmax])}\n"

    # LoCoHD statistics
    lchd_argmedian = np.argsort([x[2] for x in stat_list])[len(stat_list) // 2]
    lchd_argmin = np.argmin([x[2] for x in stat_list])
    lchd_argmax = np.argmax([x[2] for x in stat_list])

    out_str += "## Statistics for the Median LoCoHD Values\n"
    out_str += f"- mean: {np.mean([x[2] for x in stat_list]):.5f}\n"
    out_str += f"- std: {np.std([x[2] for x in stat_list]):.5f}\n\n"

    out_str += "| descriptor | structure | SpR | median LoCoHD | median lDDT |\n"
    out_str += "| ---------: | :-------: | :-: | :-----------: | :---------: |\n"
    out_str += f"| median | {data_to_table_row(stat_list[lchd_argmedian])}"
    out_str += f"| min | {data_to_table_row(stat_list[lchd_argmin])}"
    out_str += f"| max | {data_to_table_row(stat_list[lchd_argmax])}\n"

    # lDDT statistics
    lddt_argmedian = np.argsort([x[3] for x in stat_list])[len(stat_list) // 2]
    lddt_argmin = np.argmin([x[3] for x in stat_list])
    lddt_argmax = np.argmax([x[3] for x in stat_list])

    out_str += "## Statistics for the Median lDDT Values\n"
    out_str += f"- mean: {np.mean([x[3] for x in stat_list]):.5f}\n"
    out_str += f"- std: {np.std([x[3] for x in stat_list]):.5f}\n\n"

    out_str += "| descriptor | structure | SpR | median LoCoHD | median lDDT |\n"
    out_str += "| ---------: | :-------: | :-: | :-----------: | :---------: |\n"
    out_str += f"| median | {data_to_table_row(stat_list[lddt_argmedian])}"
    out_str += f"| min | {data_to_table_row(stat_list[lddt_argmin])}"
    out_str += f"| max | {data_to_table_row(stat_list[lddt_argmax])}\n"

    # Largest deviation statistics

    deviations = list()
    for structure_key in stat_dict:
        value_matrix = np.array(list(stat_dict[structure_key].values()))
        current_deviations = np.max(value_matrix, axis=0) - np.min(value_matrix, axis=0)
        deviations.append((structure_key, *current_deviations))

    spr_dev_argmax, locohd_dev_argmax, lddt_dev_argmax = np.argmax([x[1:] for x in deviations], axis=0)

    out_str += "## Statistics for the Largest Deviations\n"
    out_str += "| value label | structure | largest value deviation |\n"
    out_str += "| ----------: | :-------: | :---------------------: |\n"
    out_str += f"| SpR | {deviations[spr_dev_argmax][0]} | {deviations[spr_dev_argmax][1]:.7f} |\n"
    out_str += f"| LoCoHD | {deviations[locohd_dev_argmax][0]} | {deviations[locohd_dev_argmax][2]:.7f} |\n"
    out_str += f"| lDDT | {deviations[lddt_dev_argmax][0]} | {deviations[lddt_dev_argmax][3]:.7f} |\n\n"

    return out_str


def get_plot_alpha(lchd_scores, lddt_scores):

    area_per_tick = np.max(lchd_scores) - np.min(lchd_scores)
    area_per_tick *= np.max(lddt_scores) - np.min(lddt_scores)
    area_per_tick /= len(lddt_scores)
    area_per_tick *= 1E3

    return area_per_tick if area_per_tick < 1. else 1.


def create_plot(key: str, lchd_scores: List[float], lddt_scores: List[float], spr: float):

    fig, ax = plt.subplots()

    ax.scatter(lchd_scores, lddt_scores,
               alpha=get_plot_alpha(lchd_scores, lddt_scores),
               edgecolors="none", c="red")

    lchd_range = np.arange(0, 0.5, 0.05)
    lddt_range = np.arange(0, 1, 0.1)

    ax.set_xticks(lchd_range, labels=[f"{tick:.0%}" for tick in lchd_range])
    ax.set_yticks(lddt_range, labels=[f"{tick:.0%}" for tick in lddt_range])
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)

    ax.set_xlabel("LoCoHD score")
    ax.set_ylabel("lDDT score")
    fig.suptitle(key)

    legend_labels = list()
    legend_labels.append(f"SpR = {spr:.4f}")
    legend_labels.append(f"mean LoCoHD = {np.mean(lchd_scores):.2%}")
    legend_labels.append(f"median LoCoHD = {np.median(lchd_scores):.2%}")
    legend_labels.append(f"mean lDDT = {np.mean(lddt_scores):.2%}")
    legend_labels.append(f"median lDDT = {np.median(lddt_scores):.2%}")
    legend_handles = [Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0), ] * len(legend_labels)

    ax.legend(legend_handles, legend_labels, loc="upper right", fontsize="small", fancybox=True,
              framealpha=0.7, handlelength=0, handletextpad=0)
    fig.savefig(WORKDIR / f"{key}.png", dpi=300)

    plt.close(fig)


def main():

    # Create the primitive assigner
    primitive_assigner = PrimitiveAssigner(Path("../../primitive_typings/all_atom_with_centroid.config.json"))

    # Create the LoCoHD instance.
    w_func = WeightFunction("uniform", [3, 10])
    tag_pairing_rule = TagPairingRule({"accept_same": False})
    lchd = LoCoHD(primitive_assigner.all_primitive_types, w_func, tag_pairing_rule)

    # The values in the structures dict are also dicts. The first key refers to the
    # structure name (like T1024), while the second key is either "true" (referring to the true structure)
    # or a string of an integer ("1", "2"...), referring to the prediction id (like T1024{PREDICTOR_KEY}_1).
    # The values of this second dict are BioPython structures.
    with open(WORKDIR / f"{PREDICTOR_KEY}_structures.pickle", "rb") as f:
        structures: Dict[str, Dict[str, Structure]] = pickle.load(f)

    # For statistics collection.
    # Key1: {structure_key}{PREDICTOR_KEY}
    # Key2: {pred_id}
    # Value: (SpR, median LoCoHD, median lDDT)
    score_statistics: ScoreStatType = dict()

    # For the global histogram.
    hist_range = [[0, 0.5], [0, 1]]
    hist_bins = 100
    hist, hist_xs, hist_ys = np.histogram2d([], [], bins=hist_bins, range=hist_range)

    for structure_key in structures:

        if not os.path.exists(LDDT_TARS_PATH / f"{structure_key}.tgz"):
            continue

        # Initialize statistics dict for the current structure.
        score_statistics[structure_key] = dict()

        # Read the lDDT values for the current structure. The lddt_dict is a dict of dicts, with
        # the first dict keys being "{structure_key}{PREDICTOR_KEY}_{structure_index}" and the second dict
        # keys being "{chain_index}/{residue_index}-{residue_name}".
        lddt_dict = read_lddt_values(structure_key)

        # Transform the real structure into a list of primitive atoms and get the anchors
        # simultaneously.
        true_pra_templates = primitive_assigner.assign_primitive_structure(structures[structure_key]["true"])
        anchors = [(idx, idx) for idx, prat in enumerate(true_pra_templates) if prat.primitive_type == "Cent"]
        true_prim_atoms = list(map(prat_to_pra, true_pra_templates))

        # For each predicted structure...
        for pred_id, structure in structures[structure_key].items():

            if pred_id == "true":
                continue

            # Transform the predicted structure.
            pred_pra_templates = primitive_assigner.assign_primitive_structure(structure)
            pred_prim_atoms = list(map(prat_to_pra, pred_pra_templates))

            # Calculate LoCoHD score with distance_cutoff = 10.
            lchd_scores = lchd.from_primitives(true_prim_atoms, pred_prim_atoms, anchors, 10)

            # Collecting the lDDT scores.
            lddt_scores = list()
            key1 = f"{structure_key}{PREDICTOR_KEY}_{pred_id}"
            for anchor, _ in anchors:
                key2 = true_prim_atoms[anchor].tag
                lddt_scores.append(lddt_dict[key1][key2])

            # Calculating the Spearman's correlation coefficient
            current_spr: float = spearmanr(lchd_scores, lddt_scores).correlation

            # Updating the statistics.
            median_lchd = float(np.median(lchd_scores))
            median_lddt = float(np.median(lddt_scores))
            score_statistics[structure_key][pred_id] = (current_spr, median_lchd, median_lddt)

            # Plotting.
            create_plot(key1, lchd_scores, lddt_scores, current_spr)

            # Update histogram.
            new_hist, _, _ = np.histogram2d(lchd_scores, lddt_scores, bins=hist_bins, range=hist_range)
            hist += new_hist

            # Saving the histogram.
            fig, ax = plt.subplots()
            fig.suptitle(f"Distribution of Scores for\nContestant {PREDICTOR_KEY}")
            ax.imshow(hist[:, ::-1].T, cmap="hot")
            ticks = list(range(0, hist_bins, 10))
            ax.set_xticks(ticks, labels=[f"{hist_xs[idx]:.0%}" for idx in ticks])
            ax.set_yticks(ticks, labels=[f"{hist_ys[-idx-1]:.0%}" for idx in ticks])
            ax.set_xlabel("LoCoHD score")
            ax.set_ylabel("lDDT score")
            fig.savefig(WORKDIR / "full_hist.png", dpi=300)
            plt.close(fig)

            print(f"{key1} done...")

    with open(WORKDIR / "statistics.md", "w") as f:
        f.write(get_statistics_str(score_statistics))


if __name__ == "__main__":
    main()
