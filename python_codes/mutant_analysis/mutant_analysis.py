import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure

from scipy.stats import spearmanr

from loco_hd import (
    PrimitiveAssigner,
    LoCoHD,
    TagPairingRule,
    WeightFunction,
    PrimitiveAtomTemplate,
    PrimitiveAtom
)

PDB_KEY = "1CHO"
SOURCE_DIR = Path(f"../../data_sources/for_mutation_tests/{PDB_KEY}")
MUTANT_PDBS = SOURCE_DIR / "mutants_em"
REF_STRUCTURE = SOURCE_DIR / f"Optimized_{PDB_KEY.lower()}.pdb"

PRIMITIVE_TYPING_SCHEME = "all_atom_with_centroid.config.json"

SKEMPI_SOURCE = Path("../../data_sources/for_mutation_tests/skempi_v2.csv")
JOULE_TO_KCAL = 0.000239006

WORKDIR = Path(f"../../workdir/for_mutation_tests/{PDB_KEY}")
MM_TO_INCH = 0.0393701


def read_ddgs():

    with open(SKEMPI_SOURCE, "r") as f:
        skempi = f.read().split("\n")[1:-1]

    skempi = [
        line.split(";")
        for line in skempi
        if line.startswith(PDB_KEY)
    ]

    out_dict = dict()
    for line in skempi:

        if "," in line[1]:  # leave out multi-site mutations
            continue

        mut_id = line[1]

        # ddG = R * T * (lnK_mut - lnK_wt)
        k_value_wt = float(line[9])
        k_value_mut = float(line[7])
        temperature = float(line[13])
        ddg_value = JOULE_TO_KCAL * 8.314 * temperature * (math.log(k_value_mut) - math.log(k_value_wt))

        # update ddg lists
        ddg_list = out_dict.get(mut_id, list())
        ddg_list.append(ddg_value)
        out_dict[mut_id] = ddg_list

    # Take the average of the ddg values if multiple is present for one mutation.
    out = [(mut_id, np.mean(ddg_list)) for mut_id, ddg_list in out_dict.items()]

    return out


def prat_to_pra(prat: PrimitiveAtomTemplate) -> PrimitiveAtom:

    resi_id = prat.atom_source.source_residue
    resname = prat.atom_source.source_residue_name
    source = f"{resi_id[2]}/{resi_id[3][1]}-{resname}"
    return PrimitiveAtom(prat.primitive_type, source, prat.coordinates)


def create_plot(resi_name: str, ddg_values: np.ndarray, resi_scores: np.ndarray):

    lchd_threshold = 0.05
    corr_threshold = 0.05
    p_value_threshold = 0.05
    count_threshold = 5

    mask = resi_scores > lchd_threshold

    if np.sum(mask) < count_threshold:
        return

    corr, p_value = spearmanr(np.abs(ddg_values[mask]), resi_scores[mask])

    if abs(corr) < corr_threshold or np.isnan(corr):
        return

    if p_value > p_value_threshold:
        return

    fig, ax = plt.subplots()
    fig.set_size_inches(88 * MM_TO_INCH, 88 * MM_TO_INCH)

    ax.scatter(
        np.abs(ddg_values[mask]), resi_scores[mask],
        marker=".", color="black"
    )
    ax.set_yticks(
        ax.get_yticks(),
        labels=[f"{x:.1%}" for x in ax.get_yticks()]
    )
    ax.set_xlabel(
        "$| \\Delta \\Delta G | / \\mathrm{kcal} \\, \\mathrm{mol}^{-1}$"
    )
    ax.set_ylabel(
        "LoCoHD score"
    )

    fig.savefig(WORKDIR / f"{resi_name.replace('/', '.')}.svg", dpi=300)
    plt.close(fig)


def main():

    plt.rcParams["font.size"] = 7
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["figure.subplot.left"] = 0.2
    plt.rcParams["figure.subplot.bottom"] = 0.15

    # Reading the ddG values.
    mut_ddg_values = read_ddgs()

    # LoCoHD instance initialization.
    assigner = PrimitiveAssigner(Path("../../primitive_typings") / PRIMITIVE_TYPING_SCHEME)
    tag_pairing_rule = TagPairingRule({"accept_same": False})
    weight_function = WeightFunction("uniform", [3., 10.])
    locohd = LoCoHD(assigner.all_primitive_types, weight_function, tag_pairing_rule)

    # Reading the experimental structure.
    pdb_parser = PDBParser(QUIET=True)

    exp_structure: Structure = pdb_parser.get_structure("WT", REF_STRUCTURE)
    exp_structure_prats = assigner.assign_primitive_structure(exp_structure)
    exp_structure_pras = list(map(prat_to_pra, exp_structure_prats))
    exp_structure_anchors = [
        idx
        for idx, pra in enumerate(exp_structure_pras)
        if pra.primitive_type == "Cent"
    ]

    # Dealing with the mutant structures.
    all_lchd_scores = list()

    for mut_idx, (mut_id, _) in enumerate(mut_ddg_values):

        # Reading the mutant structures.
        mut_filename = f"{PDB_KEY.lower()}_{mut_id}_em.pdb"
        mut_structure: Structure = pdb_parser.get_structure(mut_id, MUTANT_PDBS / mut_filename)
        mut_structure_prats = assigner.assign_primitive_structure(mut_structure)
        mut_structure_pras = list(map(prat_to_pra, mut_structure_prats))
        mut_structure_anchors = [
            idx
            for idx, pra in enumerate(mut_structure_pras)
            if pra.primitive_type == "Cent"
        ]

        # LoCoHD calculation.
        anchors = list(zip(exp_structure_anchors, mut_structure_anchors))
        locohd_scores = locohd.from_primitives(
            exp_structure_pras, mut_structure_pras, anchors, 10.
        )

        all_lchd_scores.append(locohd_scores)
        print(f"{mut_idx}: {mut_filename} done!")

    all_lchd_scores = np.array(all_lchd_scores)  # (number of mutations, number of residues)
    ddg_values = np.array([x[1] for x in mut_ddg_values])  # (number of mutations, )

    # Plotting.
    for anchor_idx, resi_scores in enumerate(all_lchd_scores.T):

        pra_idx = exp_structure_anchors[anchor_idx]
        resi_name = exp_structure_pras[pra_idx].tag

        create_plot(resi_name, ddg_values, resi_scores)


if __name__ == "__main__":
    main()
