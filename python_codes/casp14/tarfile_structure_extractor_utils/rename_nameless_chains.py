from typing import Dict

from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain


def rename_nameless_chains(all_structures: Dict[str, Dict[str, Structure]]) -> None:

    for structure_name, structure_bundle in all_structures.items():

        current_chains: Dict[str, Chain] = structure_bundle["true"][0].child_dict

        print(f"{structure_name} contains {len(current_chains)} chain(s)!")

        if len(current_chains) != 1:
            continue

        chain_id, chain_obj = list(current_chains.items())[0]

        print(f"Chain ID is {chain_id}!")

        if chain_id != " ":
            continue

        for structure in list(structure_bundle.values()):
            structure[0][" "].id = "A"  # Set new chain ID
