from typing import List, Dict

from Bio.PDB.Entity import Entity
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom


def _strict_atom_filter(structures: List[Entity]):
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
        _strict_atom_filter(common_children)


def apply_filtering(all_structures: Dict[str, Dict[str, Structure]]):

    # Do the filtration.
    for structure_name, structure_bundle in all_structures.items():
        bundle_list = list(structure_bundle.values())
        _strict_atom_filter(bundle_list)

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
