from pathlib import Path
from typing import Dict
from Bio.PDB.PDBParser import PDBParser
import matplotlib.pyplot as plt
import numpy as np

from loco_hd import (
    LoCoHD,
    PrimitiveAtom,
    PrimitiveAssigner,
    PrimitiveAtomTemplate,
    WeightFunction,
    TagPairingRule
)

STRUCTURE_SOURCES = Path("../data_sources/pdb_files/ApAAP")
PROT_REF_PATH = STRUCTURE_SOURCES / "Optimized_3o4g_cleaned.pdb"
PROT_MUT_PATH = STRUCTURE_SOURCES / "Optimized_3o4h_cleaned.pdb"
WORKDIR = Path("../workdir/ApAAP")
PLOT_NAME = "3o4g_vs_3o4h"

MAX_LCHD = 0.3
MM_TO_INCH = 0.0393701


def create_histograms(
        name: str,
        lchd_scores: Dict[str, float],
        output_path: Path
):

    plt.rcParams["font.size"] = 7
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["figure.subplot.left"] = 0.2
    plt.rcParams["figure.subplot.bottom"] = 0.15

    def resi_id_to_label(resi_id: str) -> str:

        resi_num, resi_name, insertion_code = resi_id[2:].split("-")
        resi_num = resi_num if insertion_code.strip() == "" else resi_num + insertion_code

        label = "$"
        label += f"\\mathrm{{{resi_name}}}"
        label += f"^{{{resi_num}}}"
        label += "$"
        return label

    fig, ax = plt.subplots()

    first_scores = sorted(lchd_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    first_scores = list(map(
        lambda item: (resi_id_to_label(item[0]), item[1]),
        first_scores
    ))

    # Create horizontal barplots
    ax.barh(
        list(map(lambda x: x[0], first_scores)),
        list(map(lambda x: x[1], first_scores)),
        color="blue"
    )

    x_ticks_lchd = np.arange(0, MAX_LCHD + 1E-10, MAX_LCHD / 10)

    ax.set_xticks(x_ticks_lchd, labels=[f"{x:.0%}" for x in x_ticks_lchd])
    ax.set_xlabel("LoCoHD score")
    ax.set_ylabel("Residue name")

    fig.set_size_inches(88 * MM_TO_INCH, 88 * MM_TO_INCH)
    fig.subplots_adjust(left=0.2, right=0.95, wspace=0.4)
    fig.savefig(output_path / f"{name}_top10lchd.svg", dpi=300)

    plt.close(fig)


def prat_to_pra(prat: PrimitiveAtomTemplate) -> PrimitiveAtom:

    resi_id = prat.atom_source.source_residue
    resname = prat.atom_source.source_residue_name
    source = f"{resi_id[2]}/{resi_id[3][1]}-{resname}-{resi_id[3][2]}"  # WITH INSERTION CODE!
    return PrimitiveAtom(prat.primitive_type, source, prat.coordinates)


def main():

    pdb_parser = PDBParser(QUIET=True)
    protein_ref = pdb_parser.get_structure("ref", str(PROT_REF_PATH))
    protein_mut = pdb_parser.get_structure("mut", str(PROT_MUT_PATH))

    primitive_assigner = PrimitiveAssigner(Path("../primitive_typings/all_atom_with_centroid.config.json"))
    weight_function = WeightFunction("uniform", [3., 10.])
    tag_pairing_rule = TagPairingRule({"accept_same": False})
    locohd = LoCoHD(primitive_assigner.all_primitive_types, weight_function, tag_pairing_rule)

    ref_prats = primitive_assigner.assign_primitive_structure(protein_ref)
    ref_pras = list(map(prat_to_pra, ref_prats))
    mut_prats = primitive_assigner.assign_primitive_structure(protein_mut)
    mut_pras = list(map(prat_to_pra, mut_prats))

    ref_anchor_idxs = [
        idx for idx, prat in enumerate(ref_prats)
        if prat.primitive_type == "Cent"
    ]
    mut_anchor_idxs = [
        idx for idx, prat in enumerate(mut_prats)
        if prat.primitive_type == "Cent"
    ]
    all_anchor_idxs = list(zip(ref_anchor_idxs, mut_anchor_idxs))

    lchd_scores = locohd.from_primitives(ref_pras, mut_pras, all_anchor_idxs, 10.)
    lchd_scores = {
        mut_pras[anchor_idx].tag: score
        for anchor_idx, score in
        zip(mut_anchor_idxs, lchd_scores)
    }

    for resi_id, score in lchd_scores.items():
        print(f"{resi_id}: {score:.1%}")

    create_histograms(PLOT_NAME, lchd_scores, WORKDIR)

    ref_prim_pdb = primitive_assigner.generate_primitive_pdb(ref_prats)
    with open(WORKDIR / "ref_prim.pdb", "w") as f:
        f.write(ref_prim_pdb)

    mut_prim_pdb = primitive_assigner.generate_primitive_pdb(mut_prats)
    with open(WORKDIR / "mut_prim.pdb", "w") as f:
        f.write(mut_prim_pdb)


if __name__ == "__main__":
    main()
