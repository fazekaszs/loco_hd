from typing import Dict

from Bio.PDB.Structure import Structure
from Bio.PDB.PDBIO import PDBIO

from .fake_io_handler import PDBIOFakeFileHandler


def stringify_structures(all_structures: Dict[str, Dict[str, Structure]]) -> Dict[str, Dict[str, str]]:

    out = dict()
    pdb_io = PDBIO()

    for structure_name, structure_bundle in all_structures.items():

        print(f"Stringifying bundle {structure_name}...")

        out[structure_name] = dict()
        for structure_id, structure in structure_bundle.items():

            ffh = PDBIOFakeFileHandler()
            pdb_io.set_structure(structure)
            pdb_io.save(ffh)
            out[structure_name][structure_id] = ffh.content

    return out