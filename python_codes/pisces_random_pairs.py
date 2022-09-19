import math
import os
import random
import pickle
import numpy as np

from pathlib import Path
from typing import Tuple, Union, List
from time import time
from scipy import stats

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

from loco_hd import LoCoHD, PrimitiveAtom

ATOM_ID = Tuple[int, str]
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
        ["CYS:SG", "MET:SD"]]


def assign_primitive_type(resi_name: str, atom_name: str, exclude_bb: bool = False) -> Union[str, None]:

    # Backbone atoms:
    if atom_name in ["CA", "C"]:
        return "C_ali" if not exclude_bb else None
    if atom_name == "O":
        return "O_neu" if not exclude_bb else None
    if atom_name == "N":
        return "N_neu" if not exclude_bb else None
    if atom_name in TERMINAL_O:
        return "O_neg" if not exclude_bb else None

    # Sidechain atoms:
    for group_idx, atom_group in enumerate(SIDECHAIN_ATOMS):
        if f"{resi_name}:{atom_name}" in atom_group:
            return PRIMITIVE_TYPES[group_idx]

    return None


def assign_primitive_structure(model: Model) -> Tuple[List[PrimitiveAtom], List[int]]:

    primitive_sequence = list()
    anchors = list()

    resi: Residue
    for resi in model.get_residues():

        full_resi_id = resi.full_id
        resi_id = f"{full_resi_id[2]}/{full_resi_id[3][1]}-{resi.resname}"

        # Deal with the centroid primitive types
        centroid_coord = resi.center_of_mass(geometric=True)
        primitive_atom = PrimitiveAtom("Cent", resi_id, centroid_coord)
        primitive_sequence.append(primitive_atom)
        anchors.append(len(primitive_sequence) - 1)

        # Deal with the heavy atom primitive types
        atom: Atom
        for atom in resi.get_atoms():

            primitive_type = assign_primitive_type(atom.parent.get_resname(), atom.get_name(), False)
            if primitive_type is not None:

                primitive_atom = PrimitiveAtom(primitive_type, resi_id, atom.coord)
                primitive_sequence.append(primitive_atom)
            else:

                print(f"Unknown atom name: {atom.get_name()} from residue {atom.parent.get_resname()} at {resi_id}")

    return primitive_sequence, anchors


def main():

    random.seed(1994)

    pisces_path = Path("/home/fazekaszs/PycharmProjects/databases/pisces_220222")
    save_path = Path("workdir/pisces")

    # out_file_name = "results_uniform-3-10_only-hetero-contacts.pickle"
    # out_file_name = "results_uniform-3-10_all-contacts.pickle"
    out_file_name = "results_kumaraswamy-3-10-2-5_only-hetero-contacts.pickle"

    lchd = LoCoHD(PRIMITIVE_TYPES + ["Cent", ], ("kumaraswamy", [3, 10, 2, 5]))
    only_hetero_contacts = True
    upper_cutoff = 10

    pdb_files: List[str] = list(filter(lambda x: x.endswith(".pdb"), os.listdir(pisces_path)))
    random.shuffle(pdb_files)

    time_per_anchor_list = list()
    len_list = list()
    for pdb_idx in range(len(pdb_files) - 1):

        time_start = time()

        path1 = str(pisces_path / pdb_files[pdb_idx])
        path2 = str(pisces_path / pdb_files[pdb_idx + 1])

        protein1: Model = PDBParser(QUIET=True).get_structure("", path1)[0]
        protein2: Model = PDBParser(QUIET=True).get_structure("", path2)[0]

        primitive_atoms1, anchors1 = assign_primitive_structure(protein1)
        primitive_atoms2, anchors2 = assign_primitive_structure(protein2)

        if len(anchors1) < len(anchors2):
            anchors2 = random.sample(anchors2, len(anchors1))
        else:
            anchors1 = random.sample(anchors1, len(anchors2))

        anchor_pairs = [(x, y) for x, y in zip(anchors1, anchors2)]

        lchd_scores = lchd.from_primitives(primitive_atoms1, primitive_atoms2,
                                           anchor_pairs, only_hetero_contacts, upper_cutoff)

        pdb_id1 = pdb_files[pdb_idx].replace(".pdb", "")
        pdb_id2 = pdb_files[pdb_idx + 1].replace(".pdb", "")

        cumulative_results = list()
        for anchor, lchd_score in zip(anchor_pairs, lchd_scores):
            pair_id1 = f"{pdb_id1}/{primitive_atoms1[anchor[0]].id}"
            pair_id2 = f"{pdb_id2}/{primitive_atoms2[anchor[1]].id}"
            cumulative_results.append((pair_id1, pair_id2, lchd_score))

        if os.path.exists(save_path / out_file_name):
            with open(save_path / out_file_name, "rb") as f:
                cumulative_results += pickle.load(f)

        with open(save_path / out_file_name, "wb") as f:
            pickle.dump(cumulative_results, f)

        time_end = time()

        len_list.append(len(anchor_pairs))
        time_per_anchor_list.append((time_end - time_start) / len(anchor_pairs))

        if len(len_list) > 2:

            len_mean = np.mean(len_list)
            len_std = np.std(len_list)

            time_per_anchor_mean = np.mean(time_per_anchor_list)
            time_per_anchor_std = np.std(time_per_anchor_list)

            remainers = (len(pdb_files) - (pdb_idx + 1))

            eta_mean = remainers * len_mean * time_per_anchor_mean
            eta_std = math.sqrt(
                (len_std * time_per_anchor_mean) ** 2 +
                (time_per_anchor_std * len_mean) ** 2
            ) * remainers

            eta_conf_int = stats.t.ppf(0.95, len(len_list) - 2) * eta_std / math.sqrt(len(len_list))

            completed_percent = (pdb_idx + 1) / len(pdb_files)

            print(f"ETA: {eta_mean:.1f} s +/- {eta_conf_int:.1f} s, Completed: {completed_percent:.2%}")


if __name__ == "__main__":
    main()
