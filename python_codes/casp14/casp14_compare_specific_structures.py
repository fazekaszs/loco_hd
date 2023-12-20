import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO
from pathlib import Path

from Bio.PDB.Structure import Structure

# Examined structures:
# - AF2:
#     - T1064TS427_1 and _5
#     - RESI_IDX_SHIFT = 15 to adjust for PDB ID 7JTL
#     - MAX_LCHD = 0.4
# - FEIG-R2:
#     - T1074TS480_2 and _5
#     - RESI_IDX_SHIFT = 0
#     - MAX_LCHD = 0.5
#   and
#     - T1046s1_...

PREDICTOR_NAME = "TS427"
STRUCTURE_NAME = "T1064"
PREDICTED_SUFFIX1 = "1"
PREDICTED_SUFFIX2 = "5"

WORKDIR = Path(f"../../workdir/casp14/{PREDICTOR_NAME}_results")
RESI_IDX_SHIFT = 15
MAX_LCHD = 0.4
MAX_LDDT = 1.0
MM_TO_INCH = 0.0393701


def create_histograms(
        name: str,
        lchd_scores: Dict[str, float],
        lddt_scores: Dict[str, float],
        output_path: Path
):

    plt.rcParams["font.size"] = 7
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["figure.subplot.left"] = 0.2
    plt.rcParams["figure.subplot.bottom"] = 0.15

    def resi_id_to_label(resi_id: str) -> str:

        split_resi_id = resi_id[2:].split("-")
        label = "$"
        label += f"\\mathrm{{{split_resi_id[1]}}}"
        label += f"^{{{int(split_resi_id[0]) + RESI_IDX_SHIFT}}}"
        label += "$"
        return label

    fig, ax = plt.subplots(1, 2)

    first_scores = sorted(lchd_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    first_scores = [
        (resi_id_to_label(resi_id), lchd_score, lddt_scores[resi_id])
        for resi_id, lchd_score in first_scores
    ]

    # Create horizontal barplots
    ax[0].barh(
        list(map(lambda x: x[0], first_scores)),
        list(map(lambda x: x[1], first_scores)),
        color="blue"
    )
    ax[1].barh(
        list(map(lambda x: x[0], first_scores)),
        list(map(lambda x: x[2], first_scores)),
        color="red"
    )

    x_ticks_lchd = np.arange(0, MAX_LCHD + 1E-10, MAX_LCHD / 10)
    x_ticks_lddt = np.arange(0, MAX_LDDT + 1E-10, MAX_LDDT / 10)

    ax[0].set_xticks(x_ticks_lchd, labels=[f"{x:.0%}" for x in x_ticks_lchd])
    ax[1].set_xticks(x_ticks_lddt, labels=[f"{x:.0%}" for x in x_ticks_lddt])

    ax[0].set_xlabel("LoCoHD score")
    ax[1].set_xlabel("lDDT score")

    ax[0].set_ylabel("Residue name")
    ax[1].set_ylabel("Residue name")

    fig.set_size_inches(180 * MM_TO_INCH, 88 * MM_TO_INCH)
    fig.subplots_adjust(left=0.1, right=0.95, wspace=0.4)
    fig.savefig(output_path / f"{name}_top10lchd.svg", dpi=300)

    plt.close(fig)


def b_label_structure(structure: Structure, score_dict: Dict[str, float], output_filepath: Path):

    for resi_id, score_value in score_dict.items():

        resi_num = int(resi_id[2:].split("-")[0])

        atom: Atom
        for atom in structure[0][" "][resi_num].get_atoms():
            atom.bfactor = score_value

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_filepath))


def main():

    # Loading in the structures.
    with open(WORKDIR / f"{PREDICTOR_NAME}_biopython_structures.pickle", "rb") as f:
        structures: Dict[str, Dict[str, Structure]] = pickle.load(f)

    protein_pred1 = structures[STRUCTURE_NAME][PREDICTED_SUFFIX1]
    protein_pred2 = structures[STRUCTURE_NAME][PREDICTED_SUFFIX2]

    # Loading in the scores.
    with open(WORKDIR / f"{PREDICTOR_NAME}_ost_results_extended.pickle", "rb") as f:
        scores = pickle.load(f)

    lchd_values1 = scores[STRUCTURE_NAME][int(PREDICTED_SUFFIX1)]["per_residue"]["LoCoHD"]
    lddt_values1 = scores[STRUCTURE_NAME][int(PREDICTED_SUFFIX1)]["per_residue"]["lddt"]
    lchd_values2 = scores[STRUCTURE_NAME][int(PREDICTED_SUFFIX2)]["per_residue"]["LoCoHD"]
    lddt_values2 = scores[STRUCTURE_NAME][int(PREDICTED_SUFFIX2)]["per_residue"]["lddt"]

    # Target path.
    output_path = WORKDIR / f"{STRUCTURE_NAME}{PREDICTOR_NAME}_{PREDICTED_SUFFIX1}_{PREDICTED_SUFFIX2}"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Names of the predicted structures, like T1064TS427_1.
    pred1_name = f"{STRUCTURE_NAME}{PREDICTOR_NAME}_{PREDICTED_SUFFIX1}"
    pred2_name = f"{STRUCTURE_NAME}{PREDICTOR_NAME}_{PREDICTED_SUFFIX2}"

    # Plot the histograms for the first five residues, based on the largest
    # LoCoHD order.
    create_histograms(pred1_name, lchd_values1, lddt_values1, output_path)
    create_histograms(pred2_name, lchd_values2, lddt_values2, output_path)

    # Set B-factors to LoCoHD values.
    b_label_structure(protein_pred1, lchd_values1, output_path / f"{pred1_name}_lchd_blabelled.pdb")
    b_label_structure(protein_pred2, lchd_values2, output_path / f"{pred2_name}_lchd_blabelled.pdb")
    b_label_structure(protein_pred1, lddt_values1, output_path / f"{pred1_name}_lddt_blabelled.pdb")
    b_label_structure(protein_pred2, lddt_values2, output_path / f"{pred2_name}_lddt_blabelled.pdb")


if __name__ == "__main__":
    main()
