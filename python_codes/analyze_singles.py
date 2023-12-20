from pathlib import Path
from typing import Dict, Union

from Bio.PDB.Entity import Entity
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue, DisorderedResidue
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBIO import PDBIO

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
PROT_REF_PATH = STRUCTURE_SOURCES / "3o4g_pymol_cleaned.pdb"
PROT_MUT_PATH = STRUCTURE_SOURCES / "3o4j_pymol_cleaned.pdb"
WORKDIR = Path("../workdir/ApAAP")
PLOT_NAME = "3o4g_vs_3o4j"
FILTER_ATOMS = True
MUTATION_SITES = [(" ", 524, " "), ]

MAX_LCHD = 0.3
MM_TO_INCH = 0.0393701


def remove_disordered_elements(structure: Structure):

    # Searching for disordered residues.
    residue: Union[Residue, DisorderedResidue]
    disordered_resi = list()
    for residue in structure.get_residues():

        if type(residue) is DisorderedResidue:
            disordered_resi.append(residue.full_id)

    # Replacing disordered residues.
    for _, model_id, chain_id, resi_id in disordered_resi:

        current_chain: Chain = structure[model_id][chain_id]
        correct_resi_name: str = list(current_chain[resi_id].child_dict.keys())[0]

        # Direct indexing of a DisorderedResidue references atoms, so .child_dict is necessary:
        correct_resi: Residue = current_chain[resi_id].child_dict[correct_resi_name]
        resi_idx = current_chain.child_list.index(current_chain[resi_id])

        current_chain.child_list[resi_idx] = correct_resi  # replace in parent's child list
        current_chain.child_dict[resi_id] = correct_resi  # replace in parent's child dict
        correct_resi.set_parent(current_chain)  # replace child's parent

    # Searching for disordered atoms.
    atom: Union[Atom, DisorderedAtom]
    disordered_atoms = list()
    for atom in structure.get_atoms():

        current_full_id = list(atom.get_full_id())  # Convert tuple to list.
        current_full_id[-1] = current_full_id[-1][0]  # Remove altloc id.

        if type(atom) is DisorderedAtom:
            disordered_atoms.append(current_full_id)

    # Replacing disordered atoms with atoms.
    for _, model_id, chain_id, resi_id, atom_id in disordered_atoms:
        current_resi: Residue = structure[model_id][chain_id][resi_id]
        correct_atom: Atom = list(current_resi[atom_id].child_dict.values())[0]
        atom_idx = current_resi.child_list.index(current_resi[atom_id])

        current_resi.child_list[atom_idx] = correct_atom  # replace in parent's child list
        current_resi.child_dict[atom_id] = correct_atom  # replace in parent's child dictionary
        correct_atom.set_parent(current_resi)  # replace child's parent

        correct_atom.disordered_flag = 0  # set the atom's disordered flag to false
        current_resi.disordered = 0  # set the residue's disordered flag to false


def filter_atoms(ref_structure: Entity, structure: Entity):

    if type(ref_structure) is Residue:
        if ref_structure.full_id[-1] in MUTATION_SITES:  # Leave out mutation sites
            return

    for child_key in list(structure.child_dict.keys()):

        if child_key not in ref_structure.child_dict:
            structure.detach_child(child_key)
        elif type(structure.child_dict[child_key]) is not Atom:
            filter_atoms(ref_structure.child_dict[child_key], structure.child_dict[child_key])


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

    # Reading in the PDB structures.
    pdb_parser = PDBParser(QUIET=True)
    protein_ref = pdb_parser.get_structure("ref", str(PROT_REF_PATH))
    protein_mut = pdb_parser.get_structure("mut", str(PROT_MUT_PATH))

    # Filtering and saving the filtered structures.
    if FILTER_ATOMS:

        pdb_io = PDBIO()

        remove_disordered_elements(protein_ref)
        remove_disordered_elements(protein_mut)

        filter_atoms(protein_mut, protein_ref)
        filter_atoms(protein_ref, protein_mut)

        pdb_io.set_structure(protein_ref)
        filtered_ref_path = PROT_REF_PATH.parent / f"{PROT_REF_PATH.stem}_filtered.pdb"
        pdb_io.save(str(filtered_ref_path))

        pdb_io.set_structure(protein_mut)
        filtered_mut_path = PROT_MUT_PATH.parent / f"{PROT_MUT_PATH.stem}_filtered.pdb"
        pdb_io.save(str(filtered_mut_path))

    # LoCoHD instance initialization.
    primitive_assigner = PrimitiveAssigner(Path("../primitive_typings/all_atom_with_centroid.config.json"))
    weight_function = WeightFunction("uniform", [3., 10.])
    tag_pairing_rule = TagPairingRule({"accept_same": False})
    locohd = LoCoHD(primitive_assigner.all_primitive_types, weight_function, tag_pairing_rule)

    # Getting the primitive structures.
    ref_prats = primitive_assigner.assign_primitive_structure(protein_ref)
    ref_pras = list(map(prat_to_pra, ref_prats))
    mut_prats = primitive_assigner.assign_primitive_structure(protein_mut)
    mut_pras = list(map(prat_to_pra, mut_prats))

    # Setting the anchors.
    ref_anchor_idxs = [
        idx for idx, prat in enumerate(ref_prats)
        if prat.primitive_type == "Cent"
    ]
    mut_anchor_idxs = [
        idx for idx, prat in enumerate(mut_prats)
        if prat.primitive_type == "Cent"
    ]
    all_anchor_idxs = list(zip(ref_anchor_idxs, mut_anchor_idxs))

    # Calculating the LoCoHD scores.
    lchd_scores = locohd.from_primitives(ref_pras, mut_pras, all_anchor_idxs, 10.)
    lchd_scores = {
        mut_pras[anchor_idx].tag: score
        for anchor_idx, score in
        zip(mut_anchor_idxs, lchd_scores)
    }

    # Printing the LoCoHD scores.
    for resi_id, score in lchd_scores.items():
        print(f"{resi_id}: {score:.1%}")

    # Plotting the LoCoHD scores.
    create_histograms(PLOT_NAME, lchd_scores, WORKDIR)

    # Saving the primitive structures.
    ref_prim_pdb = primitive_assigner.generate_primitive_pdb(ref_prats)
    with open(WORKDIR / f"{PROT_REF_PATH.stem}_primstruct.pdb", "w") as f:
        f.write(ref_prim_pdb)

    mut_prim_pdb = primitive_assigner.generate_primitive_pdb(mut_prats)
    with open(WORKDIR / f"{PROT_MUT_PATH.stem}_primstruct.pdb", "w") as f:
        f.write(mut_prim_pdb)


if __name__ == "__main__":
    main()
