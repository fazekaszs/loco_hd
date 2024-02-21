from typing import List, Dict

from Bio.PDB.Entity import Entity
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom


def _child_dict_key_filter(structures: List[Entity]):

    # Define base case.
    if type(structures[0]) is Atom:
        return

    # Determine common child keys.
    valid_keys = set(structures[0].child_dict.keys())
    for s in structures[1:]:
        valid_keys.intersection_update(s.child_dict.keys())

    # Delete non-common children.
    for s in structures:
        for child_id in list(s.child_dict.keys()):
            if child_id not in valid_keys:
                s.detach_child(child_id)

    # Call normalization for common children recursively.
    for child_id in valid_keys:
        common_children = [s.child_dict[child_id] for s in structures]
        _child_dict_key_filter(common_children)


def apply_common_id_filtering(all_structures: Dict[str, Dict[str, Structure]]):

    # Do the filtration.
    for structure_name, structure_bundle in all_structures.items():
        bundle_list = list(structure_bundle.values())
        _child_dict_key_filter(bundle_list)

    # Report.
    no_atoms_left = set()
    print(f"Atom counts after filtration:")
    for structure_name, structure_bundle in all_structures.items():

        n_of_atoms_true = len(list(structure_bundle["true"].get_atoms()))
        if n_of_atoms_true == 0:
            print(f"No atoms left in {structure_name} after filtering!")
            no_atoms_left.add(structure_name)
            continue

        out_str = f"{structure_name}: "
        for structure_id, structure in structure_bundle.items():
            n_of_atoms = len(list(structure.get_atoms()))
            out_str += f"({structure_id}, {n_of_atoms}) "
        print(out_str)

    # Delete empty bundles.
    for structure_name in no_atoms_left:
        del all_structures[structure_name]


def _get_residue_type_id(resi: Residue) -> str:

    full_id = resi.full_id
    return f"{full_id[2]}/{full_id[3][1]}-{resi.resname}"


def apply_common_resi_filtering(all_structures: Dict[str, Dict[str, Structure]]):

    for structure_name, structure_bundle in all_structures.items():

        print(f"Searching for common residues in {structure_name}...")

        # Initialize common residue dictionary.
        common_resi_dict = {
            chain_id: set(map(_get_residue_type_id, chain.get_residues()))
            for chain_id, chain in
            structure_bundle["true"][0].child_dict.items()
        }

        # This is needed for reporting.
        original_resi_counts = {
            chain_id: len(common_resi)
            for chain_id, common_resi
            in common_resi_dict.items()
        }

        # Update common residue dictionary with each structure in the bundle.
        for structure_id, structure in structure_bundle.items():

            if structure_id == "true":
                continue

            for chain_id, chain in structure[0].child_dict.items():
                available_residues = set(map(_get_residue_type_id, chain.get_residues()))
                common_resi_dict[chain_id].intersection_update(available_residues)

        # Reporting changes in residue counts.
        out_str = "Chain residue count changes: "
        for chain_id, original_count in original_resi_counts.items():
            out_str += f"{chain_id}: {original_count} -> {len(common_resi_dict[chain_id])}, "
        out_str = out_str[:-2]
        print(out_str)

        # Remove residues that are not common.
        for chain_id, common_residues in common_resi_dict.items():
            for structure_id, structure in structure_bundle.items():

                # Skip predicted structures that do not contain the current
                #  reference structure chain name.
                if chain_id not in structure[0].child_dict:
                    continue

                n_of_removed_resi = 0
                current_chain = structure[0][chain_id]
                chain_residues = list(current_chain.child_dict.values())
                for residue in chain_residues:

                    residue_type_id = _get_residue_type_id(residue)
                    if residue_type_id not in common_residues:
                        n_of_removed_resi += 1
                        current_chain.detach_child(residue.full_id[-1])

                print(
                    f"Removed {n_of_removed_resi} non common residues from "
                    f"{structure_name}_{structure_id} in chain {chain_id}."
                )
