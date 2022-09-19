import tarfile
import codecs
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

from pathlib import Path
from typing import List, Dict, Tuple, Union
from loco_hd import LoCoHD, PrimitiveAtom
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

FULL_ID_TYPE = Tuple[str, int, str, Tuple[str, int, str], Tuple[str, str]]
TERMINAL_O = ["OT1", "OT2", "OC1", "OC2", "OXT"]
PRIMITIVE_TYPES = ["O_neg", "O_neu", "N_pos", "N_neu", "C_ali", "C_aro", "S"]
SIDECHAIN_ATOMS = [
        ["GLU:OE1", "GLU:OE2", "ASP:OD1", "ASP:OD2"],
        ["GLN:OE1", "ASN:OD1", "SER:OG", "THR:OG1", "TYR:OH"],
        ["ARG:NE", "ARG:NH1", "ARG:NH2", "LYS:NZ"],
        ["GLN:NE2", "ASN:ND2", "HIS:ND1", "HIS:NE2", "TRP:NE1"],
        ["ALA:CB", "VAL:CB", "VAL:CG1", "VAL:CG2", "ILE:CB", "ILE:CG1",
         "ILE:CG2", "ILE:CD1", "ILE:CD", "LEU:CB", "LEU:CG", "LEU:CD1",
         "LEU:CD2", "PHE:CB", "SER:CB", "THR:CB", "THR:CG2", "ASP:CB",
         "ASP:CG", "ASN:CB", "ASN:CG", "GLU:CB", "GLU:CG", "GLU:CD",
         "GLN:CB", "GLN:CG", "GLN:CD", "ARG:CB", "ARG:CG", "ARG:CD",
         "ARG:CE", "ARG:CZ", "LYS:CB", "LYS:CG", "LYS:CD", "LYS:CE",
         "HIS:CB", "CYS:CB", "MET:CB", "MET:CG", "MET:CE", "PRO:CB",
         "PRO:CG", "PRO:CD", "TYR:CB", "TRP:CB"],
        ["HIS:CG", "HIS:CD2", "HIS:CE1", "PHE:CG", "PHE:CD1", "PHE:CD2",
         "PHE:CE1", "PHE:CE2", "PHE:CZ", "TYR:CG", "TYR:CD1", "TYR:CD2",
         "TYR:CE1", "TYR:CE2", "TYR:CZ", "TRP:CG", "TRP:CD1", "TRP:CD2",
         "TRP:CE2", "TRP:CE3", "TRP:CZ2", "TRP:CZ3", "TRP:CH2"],
        ["CYS:SG", "MET:SD"]
]


def assign_primitive_type(resi_name: str, atom_name: str) -> Union[str, None]:

    # Backbone atoms:
    if atom_name in ["CA", "C"]:
        return "C_ali"
    if atom_name == "O":
        return "O_neu"
    if atom_name == "N":
        return "N_neu"
    if atom_name in TERMINAL_O:
        return "O_neg"

    # Sidechain atoms:
    for group_idx, atom_group in enumerate(SIDECHAIN_ATOMS):
        if f"{resi_name}:{atom_name}" in atom_group:
            return PRIMITIVE_TYPES[group_idx]

    return None


def resi_id_seq_from_structure(structure: Structure):

    id_seq = list()

    resi: Residue
    for resi in structure.get_residues():
        id_seq.append(resi.get_full_id())
    return id_seq


def get_primitive_sequence(structure: Structure, resi_id_seq: List):

    prim_atoms = list()
    anchors = list()

    for resi_id in resi_id_seq:

        resi: Residue = structure[resi_id[1]][resi_id[2]][resi_id[3]]

        prim_atom_source = f"{resi_id[2]}/{resi_id[3][1]}-{resi.get_resname()}"

        prim_atom = PrimitiveAtom("Cent", prim_atom_source, resi.center_of_mass(geometric=True))
        prim_atoms.append(prim_atom)
        anchors.append(len(prim_atoms) - 1)

        atom: Atom
        for atom in resi.get_atoms():

            prim_type = assign_primitive_type(resi.get_resname(), atom.get_name())
            prim_atom = PrimitiveAtom(prim_type, prim_atom_source, atom.coord)
            prim_atoms.append(prim_atom)

    return prim_atoms, anchors


def format_results(lchd_scores: List[float], primitive_atoms: List[PrimitiveAtom], anchors: List):

    for score, anchor in zip(lchd_scores, anchors):

        print(f"{primitive_atoms[anchor[0]].id}:\t{score}")


def lddt_from_text(text: str):

    text = text.split("\n")
    header_text = "Chain	ResName	ResNum	Asses.	Q.Prob.	Score"
    start_idx = next(idx for idx in range(len(text)) if text[idx].startswith(header_text))
    text = text[start_idx + 1:]
    text = list(map(lambda line: line.split("\t"), text))
    text = {f"{line[0]}/{line[2]}-{line[1]}": float(line[5]) for line in text if len(line) != 1 and line[5] != "-"}
    return text


def get_plot_alpha(lchd_scores, lddt_scores):

    area_per_tick = np.max(lchd_scores) - np.min(lchd_scores)
    area_per_tick *= np.max(lddt_scores) - np.min(lddt_scores)
    area_per_tick /= len(lddt_scores)
    area_per_tick *= 1E3

    return area_per_tick if area_per_tick < 1. else 1.


def main():

    # AF2: TS427
    # BAKER: TS473
    # BAKER-experimental: TS403
    # FEIG-R2: TS480
    # Zhang: TS129
    predictor_key = "TS427"

    lchd = LoCoHD(PRIMITIVE_TYPES + ["Cent", ], ("uniform", [3, 10]))
    lddt_tars_path = Path("/home/fazekaszs/CoreDir/PhD/PDB/casp14/lDDTs")
    workdir = Path(f"./workdir/{predictor_key}_results")
    fig, ax = plt.subplots()

    # The values in the structures dict are lists of structures, where the first structure
    # in the lists is the true structure, and the rest of them are the predicted structures.
    with open(workdir / f"{predictor_key}_structures.pickle", "rb") as f:
        structures: Dict[str, List[Structure]] = pickle.load(f)

    # For statistics collection.
    spr_values = list()
    median_lddts = list()
    median_lchds = list()

    for structure_key in structures:

        if not os.path.exists(lddt_tars_path / f"{structure_key}.tgz"):
            continue

        # Read the lDDT values for the current structure. The lddt_dict is a dict of dicts, with
        # the first dict keys being "{structure_key}{predictor_key}_{structure_index}" and the second dict
        # keys being "{chain_index}/{residue_index}-{residue_name}".
        lddt_dict = dict()
        with tarfile.open(lddt_tars_path / f"{structure_key}.tgz") as tf:
            tf_members = tf.getmembers()
            tf_members = list(filter(lambda m: f"{structure_key}{predictor_key}_" in m.name, tf_members))
            for tf_member in tf_members:
                f = tf.extractfile(tf_member)
                content = codecs.getreader("utf-8")(f).read()
                content = lddt_from_text(content)
                member_key = tf_member.name.split("/")[2].replace(".lddt", "")
                lddt_dict[member_key] = content
                f.close()

        # Transform the real structure into a list of primitive atoms and get the anchors
        # simultaneously.
        resi_ids = resi_id_seq_from_structure(structures[structure_key][0])
        true_prim_atoms, anchors = get_primitive_sequence(structures[structure_key][0], resi_ids)
        anchors = [(x, x) for x in anchors]

        # For each predicted structure...
        for pred_idx, structure in enumerate(structures[structure_key][1:]):

            # Transform the predicted structures. The anchors are the same, so we don't need them.
            pred_prim_atoms, _ = get_primitive_sequence(structure, resi_ids)
            # Calculate LoCoHD (only_hetero_contacts = True, distance_cutoff = 10).
            lchd_scores = lchd.from_primitives(true_prim_atoms, pred_prim_atoms, anchors, True, 10)

            # Collecting the lDDT scores.
            lddt_scores = list()
            key1 = f"{structure_key}{predictor_key}_{pred_idx + 1}"
            for (anchor, _) in anchors:
                key2 = true_prim_atoms[anchor].id
                lddt_scores.append(lddt_dict[key1][key2])

            # Updating the statistics.
            spr_values.append(spearmanr(lchd_scores, lddt_scores).correlation)
            median_lchds.append(np.median(lchd_scores))
            median_lddts.append(np.median(lddt_scores))

            # Plotting.
            ax.cla()
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
            fig.suptitle(key1)
            legend_handles = [Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0), ] * 3
            legend_labels = list()
            legend_labels.append(f"SpR = {spr_values[-1]:.5f}")
            legend_labels.append(f"Avg LoCoHD = {np.mean(lchd_scores):.1%}")
            legend_labels.append(f"Avg lDDT = {np.mean(lddt_scores):.1%}")
            ax.legend(legend_handles, legend_labels,
                      loc="upper right", fontsize="small", fancybox=True,
                      framealpha=0.7, handlelength=0, handletextpad=0)
            fig.savefig(workdir / f"{key1}.png", dpi=300)

            print(f"{key1} done...")

    # Saving statistics.
    out_str = ""
    out_str += f"Mean SpR: {np.mean(spr_values)}\n"
    out_str += f"Median SpR: {np.median(spr_values)}\n"
    out_str += f"Std SpR: {np.std(spr_values)}\n"
    out_str += f"Min SpR: {np.min(spr_values)}\n"
    out_str += f"Max SpR: {np.max(spr_values)}\n"

    out_str += f"Mean median lDDT: {np.mean(median_lddts)}\n"
    out_str += f"Median median lDDT: {np.median(median_lddts)}\n"
    out_str += f"Std median lDDT: {np.std(median_lddts)}\n"
    out_str += f"Min median lDDT: {np.min(median_lddts)}\n"
    out_str += f"Max median lDDT: {np.max(median_lddts)}\n"

    out_str += f"Mean median LoCoHD: {np.mean(median_lchds)}\n"
    out_str += f"Median median LoCoHD: {np.median(median_lchds)}\n"
    out_str += f"Std median LoCoHD: {np.std(median_lchds)}\n"
    out_str += f"Min median LoCoHD: {np.min(median_lchds)}\n"
    out_str += f"Max median LoCoHD: {np.max(median_lchds)}\n"

    with open(workdir / "statistics.txt", "w") as f:
        f.write(out_str)


if __name__ == "__main__":
    main()
