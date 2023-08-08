import os
import numpy as np
import tarfile
import codecs
import matplotlib.pyplot as plt
from typing import Dict, List
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO
from pathlib import Path
from loco_hd import LoCoHD, PrimitiveAtom, PrimitiveAssigner, PrimitiveAtomTemplate, WeightFunction, TagPairingRule
from casp14_predictor_extractor import filter_atoms

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
PREDICTED_SUFFIX1 = "_1"
PREDICTED_SUFFIX2 = "_5"

LDDT_TARS_PATH = Path("../data_sources/casp14/lDDTs")
PDB_DIR_PATH = Path(f"../data_sources/casp14_{STRUCTURE_NAME}/pdbs")
TARGET_PATH = Path("../workdir/casp14")
RESI_IDX_SHIFT = 15
MAX_LCHD = 0.4


def lddt_from_text(text: str):

    text = text.split("\n")
    header_text = "Chain	ResName	ResNum	Asses.	Q.Prob.	Score"
    start_idx = next(idx for idx in range(len(text)) if text[idx].startswith(header_text))
    text = text[start_idx + 1:]
    text = list(map(lambda line: line.split("\t"), text))
    text = {f"{line[0]}/{line[2]}-{line[1]}": float(line[5]) for line in text if len(line) != 1 and line[5] != "-"}
    return text


def read_lddt_values() -> Dict[str, Dict[str, float]]:

    lddt_dict = dict()
    with tarfile.open(LDDT_TARS_PATH / f"{STRUCTURE_NAME}.tgz") as tf:
        tf_members = tf.getmembers()
        tf_members = list(filter(lambda m: f"{STRUCTURE_NAME}{PREDICTOR_NAME}_" in m.name, tf_members))
        for tf_member in tf_members:
            f = tf.extractfile(tf_member)
            content = codecs.getreader("utf-8")(f).read()
            content = lddt_from_text(content)
            member_key = tf_member.name.split("/")[2].replace(".lddt", "")
            lddt_dict[member_key] = content
            f.close()

    return lddt_dict


def prat_to_pra(prat: PrimitiveAtomTemplate) -> PrimitiveAtom:

    resi_id = prat.atom_source.source_residue
    resname = prat.atom_source.source_residue_name
    source = f"{resi_id[2]}/{resi_id[3][1]}-{resname}"
    return PrimitiveAtom(prat.primitive_type, source, prat.coordinates)


def create_lchd_histograms(
        name: str,
        lchd_scores: np.ndarray,
        anchor_idxs: List[int],
        residues: List[PrimitiveAtomTemplate],
        output_path: Path
):

    fig, ax = plt.subplots()
    first_idxs = np.argsort(lchd_scores)[::-1][:10]
    anchor_prats = [residues[anchor_idxs[idx]] for idx in first_idxs]

    # Set the y-axis tick labels to the residue names.
    labels = list()
    prat: PrimitiveAtomTemplate
    for prat in anchor_prats:

        current_label = "$"
        current_label += f"\\mathrm{{{prat.atom_source.source_residue_name}}}"
        current_label += f"^{{{prat.atom_source.source_residue[3][1] + RESI_IDX_SHIFT}}}"
        current_label += "$"
        labels.append(current_label)

    ax.barh(labels, lchd_scores[first_idxs])  # Horizontal barplot
    x_ticks = np.arange(0, MAX_LCHD, MAX_LCHD / 10)
    ax.set_xticks(x_ticks, labels=[f"{x:.0%}" for x in x_ticks])
    ax.set_xlabel("LoCoHD score")
    ax.set_ylabel("Residue name")
    fig.suptitle(f"Top 10 Largest per-Residue\nLoCoHD Values in {name}")
    plt.tight_layout()
    fig.savefig(output_path / f"{name}_top10lchd.png", dpi=300, bbox_inches="tight")

    plt.close(fig)


def main():

    # Names of the predicted structures, like T1064TS427_1.
    pred1_name = f"{STRUCTURE_NAME}{PREDICTOR_NAME}{PREDICTED_SUFFIX1}"
    pred2_name = f"{STRUCTURE_NAME}{PREDICTOR_NAME}{PREDICTED_SUFFIX2}"

    # Source paths.
    pdb_true_path = PDB_DIR_PATH / f"{STRUCTURE_NAME}.pdb"
    pdb_pred1_path = PDB_DIR_PATH / f"{pred1_name}.pdb"
    pdb_pred2_path = PDB_DIR_PATH / f"{pred2_name}.pdb"

    # Target path.
    output_path = TARGET_PATH / f"{STRUCTURE_NAME}{PREDICTOR_NAME}{PREDICTED_SUFFIX1}{PREDICTED_SUFFIX2}"
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Reading the PDB files.
    protein_true = PDBParser(QUIET=True).get_structure("true", pdb_true_path)
    protein_pred1 = PDBParser(QUIET=True).get_structure("pred1", pdb_pred1_path)
    protein_pred2 = PDBParser(QUIET=True).get_structure("pred2", pdb_pred2_path)

    filter_atoms(protein_true, protein_pred1)
    filter_atoms(protein_true, protein_pred2)

    # Create the PrimitiveAssigner instance
    primitive_assigner = PrimitiveAssigner(Path("../primitive_typings/all_atom_with_centroid.config.json"))

    # Initialize LoCoHD instance
    weight_function = WeightFunction("uniform", [3, 10])
    tag_pairing_rule = TagPairingRule({"accept_same": False})
    lchd = LoCoHD(primitive_assigner.all_primitive_types, weight_function, tag_pairing_rule)

    # Parse the structures
    pra_templates_true = primitive_assigner.assign_primitive_structure(protein_true)
    pra_templates_pred1 = primitive_assigner.assign_primitive_structure(protein_pred1)
    pra_templates_pred2 = primitive_assigner.assign_primitive_structure(protein_pred2)

    # Get the anchor indices
    anchor_pairs = [
        (idx, idx)
        for idx, pra_template in enumerate(pra_templates_true)
        if pra_template.primitive_type == "Cent"
    ]

    # Create the PrimitiveAtom lists
    pra_true = list(map(prat_to_pra, pra_templates_true))
    pra_pred1 = list(map(prat_to_pra, pra_templates_pred1))
    pra_pred2 = list(map(prat_to_pra, pra_templates_pred2))

    # Calculate the LoCoHD values
    lchd_values1 = lchd.from_primitives(pra_true, pra_pred1, anchor_pairs, 10.)
    lchd_values1 = np.array(lchd_values1)
    lchd_values2 = lchd.from_primitives(pra_true, pra_pred2, anchor_pairs, 10.)
    lchd_values2 = np.array(lchd_values2)

    lchd_mean1 = np.mean(lchd_values1)
    print(f"Mean LoCoHD value for {pred1_name}: {lchd_mean1:.1%}")
    lchd_mean2 = np.mean(lchd_values2)
    print(f"Mean LoCoHD value for {pred2_name}: {lchd_mean2:.1%}")

    # Plot the histograms for the first five residues, based on the largest
    # LoCoHD order
    anchor_idxs = [x[0] for x in anchor_pairs]
    create_lchd_histograms(pred1_name, lchd_values1, anchor_idxs, pra_templates_pred1, output_path)
    create_lchd_histograms(pred2_name, lchd_values2, anchor_idxs, pra_templates_pred2, output_path)

    # Save the primitive structures
    pdb_str_true = primitive_assigner.generate_primitive_pdb(pra_templates_true)
    with open(output_path / f"{STRUCTURE_NAME}{PREDICTOR_NAME}_true_primitiveStructure.pdb", "w") as f:
        f.write(pdb_str_true)

    pdb_str_pred1 = primitive_assigner.generate_primitive_pdb(pra_templates_pred1)
    with open(output_path / f"{pred1_name}_primitiveStructure.pdb", "w") as f:
        f.write(pdb_str_pred1)

    pdb_str_pred2 = primitive_assigner.generate_primitive_pdb(pra_templates_pred2)
    with open(output_path / f"{pred2_name}_primitiveStructure.pdb", "w") as f:
        f.write(pdb_str_pred2)

    # Set B-factors to LoCoHD values
    for idx, (anchor_idx, _) in enumerate(anchor_pairs):

        resi_id = pra_templates_true[anchor_idx].atom_source.source_residue

        atom: Atom
        for atom in protein_pred1[resi_id[1]][resi_id[2]][resi_id[3]].get_atoms():
            atom.bfactor = lchd_values1[idx]

        for atom in protein_pred2[resi_id[1]][resi_id[2]][resi_id[3]].get_atoms():
            atom.bfactor = lchd_values2[idx]

    io = PDBIO()

    io.set_structure(protein_pred1)
    io.save(str(output_path / f"{pred1_name}_lchd_blabelled.pdb"))

    io.set_structure(protein_pred2)
    io.save(str(output_path / f"{pred2_name}_lchd_blabelled.pdb"))

    # Load lDDT values
    lddt_values = read_lddt_values()

    # Set B-factors to lDDT values
    for idx, (anchor_idx, _) in enumerate(anchor_pairs):

        resi_id = pra_templates_true[anchor_idx].atom_source.source_residue
        resi_name = pra_templates_true[anchor_idx].atom_source.source_residue_name
        lddt_key = f"{resi_id[2]}/{resi_id[3][1]}-{resi_name}"

        atom: Atom
        for atom in protein_pred1[resi_id[1]][resi_id[2]][resi_id[3]].get_atoms():
            atom.bfactor = lddt_values[pred1_name][lddt_key]

        for atom in protein_pred2[resi_id[1]][resi_id[2]][resi_id[3]].get_atoms():
            atom.bfactor = lddt_values[pred2_name][lddt_key]

    io.set_structure(protein_pred1)
    io.save(str(output_path / f"{pred1_name}_lddt_blabelled.pdb"))

    io.set_structure(protein_pred2)
    io.save(str(output_path / f"{pred2_name}_lddt_blabelled.pdb"))


if __name__ == "__main__":
    main()
