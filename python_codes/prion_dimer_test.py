import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Union, Tuple

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO, Select

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


class AtomSelector(Select):

    def __init__(self, accepted_ids: List[str]):
        super().__init__()
        self.accepted_ids = accepted_ids

    def accept_atom(self, atom: Atom):

        resi: Residue = atom.parent
        resi_id = f"{resi.parent.id}/{resi.id[1]}"
        if resi_id in self.accepted_ids:
            return True
        return False


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
        resi_id = f"{full_resi_id[2]}/{full_resi_id[3][1]}"

        # Deal with the centroid primitive types
        centroid_coord = resi.center_of_mass(geometric=True)
        primitive_atom = PrimitiveAtom("Cent", resi_id, centroid_coord)
        primitive_sequence.append(primitive_atom)
        anchors.append(len(primitive_sequence) - 1)

        print(f"Resi id: {resi_id}")

        # Deal with the heavy atom primitive types
        atom: Atom
        for atom in resi.get_atoms():

            primitive_type = assign_primitive_type(atom.parent.get_resname(), atom.get_name(), True)
            if primitive_type is not None:

                primitive_atom = PrimitiveAtom(primitive_type, resi_id, atom.coord)
                primitive_sequence.append(primitive_atom)
            else:

                print(f"Unknown atom name: {atom.get_name()} from residue {atom.parent.get_resname()} at {resi_id}")

    return primitive_sequence, anchors


def main():

    prot1_path = Path("./workdir/pdb_files/human_prions/1qm3_hPri_WT.pdb")
    prot2_path = Path("./workdir/pdb_files/human_prions/1i4m_hPri_WT_dimer_correct.pdb")

    # Read proteins
    prot1_structure = PDBParser(QUIET=True).get_structure("", prot1_path)[0]
    prot2_structure = PDBParser(QUIET=True).get_structure("", prot2_path)[0]

    primitive_sequence1, anchors1 = assign_primitive_structure(prot1_structure)
    primitive_sequence2, anchors2 = assign_primitive_structure(prot2_structure)

    # Shape the anchors, so the one-to-one correspondence is fulfilled
    anchors1 = anchors1[:-2]
    anchors2 = anchors2[:len(anchors2) // 2]
    anchors2 = anchors2[6:]
    anchors = [(a1, a2) for a1, a2 in zip(anchors1, anchors2)]
    lchd = LoCoHD(PRIMITIVE_TYPES + ["Cent", ], ("kumaraswamy", [3, 10, 2, 5]))

    lchd_scores = lchd.from_primitives(primitive_sequence1, primitive_sequence2, anchors, True, 10)
    lchd_scores = np.array(lchd_scores)
    fig, ax = plt.subplots()
    plt.tight_layout()

    skip_labels = 10
    x_ax_labels = [primitive_sequence1[idx].id.split("/")[1] for idx in anchors1[::skip_labels]]
    x_ax_ticks = list(range(0, len(anchors1), skip_labels))

    ax.plot(lchd_scores)
    ax.set_xticks(x_ax_ticks, labels=x_ax_labels)
    ax.set_yticklabels([f"{x:.1%}" for x in ax.get_yticks()])
    ax.set_xlabel("Residue number")
    ax.set_ylabel("LoCoHD score")

    fig.savefig("./workdir/prion_dimer_test.png", dpi=300, bbox_inches="tight")

    # Save b-factor labelled structure
    pdb_io = PDBIO()

    for anchor, score in zip(anchors2, lchd_scores):

        resi_id = primitive_sequence2[anchor].id.split("/")

        atom: Atom
        for atom in prot2_structure[resi_id[0]][int(resi_id[1])].get_atoms():
            atom.bfactor = score

    pdb_io.set_structure(prot2_structure)
    accepted_atoms = [primitive_sequence2[anchor].id for anchor in anchors2]
    pdb_io.save(f"./workdir/prion_dimer_blabelled.pdb", select=AtomSelector(accepted_atoms))


if __name__ == "__main__":
    main()
