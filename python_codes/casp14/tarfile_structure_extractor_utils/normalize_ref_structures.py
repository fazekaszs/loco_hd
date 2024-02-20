from typing import Union, Dict

from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue, DisorderedResidue
from Bio.PDB.Atom import Atom, DisorderedAtom

_RESI_TLCS = {
    "GLY", "ALA", "VAL", "ILE", "LEU",
    "PHE", "SER", "THR", "TYR", "ASP",
    "GLU", "ASN", "GLN", "CYS", "MET",
    "PRO", "LYS", "ARG", "TRP", "HIS"
}


def _remove_disordered_elements(structure: Structure) -> None:

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


def _remove_non_canonical_amino_acids(structure: Structure, structure_name: str) -> None:

    residues = list(structure.get_residues())

    for resi in residues:

        if resi.resname in _RESI_TLCS:
            continue

        print(f"Removing non-canonical amino acid {resi.resname} from reference structure {structure_name}!")
        resi.parent.detach_child(resi.full_id[-1])


def normalize_ref_structures(all_structures: Dict[str, Dict[str, Structure]]) -> None:

    for structure_name, structure_bundle in all_structures.items():

        ref_structure = structure_bundle["true"]
        _remove_disordered_elements(ref_structure)
        _remove_non_canonical_amino_acids(ref_structure, structure_name)

        print(f"Reference structure {structure_name} normalized!")