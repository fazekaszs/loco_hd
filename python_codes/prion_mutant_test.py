import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Union, Tuple

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Chain import Chain
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


def assign_primitive_structure(chain: Chain) -> Tuple[List[PrimitiveAtom], List[int]]:

    primitive_sequence = list()
    anchors = list()

    resi: Residue
    for resi in chain.get_residues():

        # Deal with the centroid primitive types
        resi_id = str(resi.get_id()[1])
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


def compare_structures(prot1_path: Path, prot2_path: Path, save_name: str, save_path: Path):

    # Read proteins
    prot1_structure = PDBParser(QUIET=True).get_structure("", prot1_path)[0].child_list[0]
    prot2_structure = PDBParser(QUIET=True).get_structure("", prot2_path)[0].child_list[0]

    # Collect primitive atom types
    atom: Atom
    primitive_sequence1, anchors1 = assign_primitive_structure(prot1_structure)
    primitive_sequence2, anchors2 = assign_primitive_structure(prot2_structure)
    print(f"Number of primitive atoms in protein 1: {len(primitive_sequence1)}")
    print(f"Number of anchor atoms in protein 1: {len(anchors1)}")
    print(f"Number of primitive atoms in protein 2: {len(primitive_sequence2)}")
    print(f"Number of anchor atoms in protein 2: {len(anchors2)}")

    # Initialize LoCoHD instance
    anchors = [(a1, a2) for a1, a2 in zip(anchors1, anchors2)]
    lchd = LoCoHD(PRIMITIVE_TYPES + ["Cent", ], ("kumaraswamy", [3, 10, 2, 5]))

    lchd_scores_only_hetero = lchd.from_primitives(primitive_sequence1, primitive_sequence2, anchors, True, 10)
    lchd_scores_only_hetero = np.array(lchd_scores_only_hetero)
    lchd_scores_all = lchd.from_primitives(primitive_sequence1, primitive_sequence2, anchors, False, 10)
    lchd_scores_all = np.array(lchd_scores_all)

    fig, ax = plt.subplots()
    plt.tight_layout()

    skip_labels = 10
    x_ax_labels = [primitive_sequence1[idx].id for idx in anchors1[::skip_labels]]
    x_ax_ticks = list(range(0, len(anchors1), skip_labels))

    ax.plot(lchd_scores_only_hetero, label="only hetero contacts")
    ax.plot(lchd_scores_all, label="all contacts")
    ax.legend(loc="upper left")
    ax.set_xticks(x_ax_ticks, labels=x_ax_labels)
    ax.set_yticklabels([f"{x:.1%}" for x in ax.get_yticks()])
    ax.set_xlabel("Residue number")
    ax.set_ylabel("LoCoHD score")

    fig.savefig(save_path / save_name, dpi=300, bbox_inches="tight")


def main():

    prot_root_path = Path("./workdir/pdb_files/human_prions")
    save_path = Path("./workdir/prion_results")

    compare_structures(prot_root_path / "1e1g_hPri_M166V.pdb",
                       prot_root_path / "1qm3_hPri_WT.pdb",
                       "1e1g_vs_1qm3.kumaraswamy-3-10-2-5.noBB.png", save_path)

    compare_structures(prot_root_path / "1e1p_hPri_S170N.pdb",
                       prot_root_path / "1qm3_hPri_WT.pdb",
                       "1e1p_vs_1qm3.kumaraswamy-3-10-2-5.noBB.png", save_path)

    compare_structures(prot_root_path / "1e1u_hPri_R220K.pdb",
                       prot_root_path / "1qm3_hPri_WT.pdb",
                       "1e1u_vs_1qm3.kumaraswamy-3-10-2-5.noBB.png", save_path)

    compare_structures(prot_root_path / "1h0l_hPri_M166C_E221C_trunc.pdb",
                       prot_root_path / "1qm3_hPri_WT.pdb",
                       "1h0l_vs_1qm3.kumaraswamy-3-10-2-5.noBB.png", save_path)


if __name__ == "__main__":
    main()
