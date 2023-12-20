import re
import json
import numpy as np

from pathlib import Path
from typing import Union, List, Tuple, Dict
from dataclasses import dataclass

from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

AtomHolders = Union[Structure, Model, Chain]
ResiFullIdType = Tuple[str, int, str, Tuple[str, int, str]]  # structure id, model id, chain id, residue id


@dataclass
class PrimitiveAtomSource:
    """
        Contains the residue source for the primitive atom, as well as the atomic-level details.
        The source_atom field contains info about how the primitive atom was constructed; it is
        either a list with only one element (denoting one source atom name), or a list with
        multiple elements (denoting that the primitive atom is a centroid of several atoms).
    """

    source_residue: ResiFullIdType
    source_residue_name: str
    source_atom: List[str]


@dataclass
class PrimitiveAtomTemplate:
    """
        Serves as an intermediate container between the original atom class (Bio.PDB.Atom) and
        the LoCoHD primitive atom (locohd.PrimitiveAtom).
    """

    primitive_type: str
    coordinates: np.ndarray
    atom_source: PrimitiveAtomSource


@dataclass
class TypingSchemeElement:

    primitive_type: str
    residue_matcher: re.Pattern
    atom_matcher: re.Pattern
    atom_counter: Union[int, str]

    def match_resi(self, resi_name: str) -> bool:
        return self.residue_matcher.fullmatch(resi_name) is not None

    def match_atom(self, atom_name: str) -> bool:
        return self.atom_matcher.fullmatch(atom_name) is not None


class PrimitiveAssigner:
    """
        Aims to contain the primitive typing scheme and is able to convert protein structures (i.e.,
        PDB files) to a list of PrimitiveAtomTemplates (see the assign_primitive_strucutre method).
    """

    def __init__(self, config_path: Path):

        config_type = Dict[str, List[Tuple[str, str, Union["any", int]]]]
        with open(config_path, "r") as f:
            config: config_type = json.load(f)

        self.scheme: List[TypingSchemeElement] = list()
        for primitive_type, scheme_elements in config.items():

            for element in scheme_elements:

                c_residue_matcher = re.compile(element[0])
                c_atom_matcher = re.compile(element[1])
                atom_counter = 1 if len(element) == 2 else element[2]

                new_element = TypingSchemeElement(primitive_type, c_residue_matcher, c_atom_matcher, atom_counter)
                self.scheme.append(new_element)

    @property
    def all_primitive_types(self) -> List[str]:
        return list({element.primitive_type for element in self.scheme})

    @all_primitive_types.setter
    def all_primitive_types(self, value):
        raise Exception("Cannot set all_primitive_types directly, since it depends on the config file!")

    def assign_primitive_structure(self, structure: AtomHolders) -> List[PrimitiveAtomTemplate]:

        out = list()

        resi: Residue
        for resi in structure.get_residues():

            resi_name = resi.resname
            resi_id = resi.full_id

            for tse in self.scheme:

                if not tse.match_resi(resi_name):
                    continue

                atom_names = list()
                atom_coords = list()

                atom: Atom
                for atom in resi.get_atoms():

                    if not tse.match_atom(atom.name):
                        continue

                    atom_names.append(atom.name)
                    atom_coords.append(atom.coord)

                if tse.atom_counter == "any":
                    pass
                elif tse.atom_counter == len(atom_coords):
                    pass
                else:
                    continue

                centroid = np.mean(atom_coords, axis=0)
                pras = PrimitiveAtomSource(resi_id, resi_name, atom_names)
                prat = PrimitiveAtomTemplate(tse.primitive_type, centroid, pras)
                out.append(prat)

        return out

    def generate_primitive_pdb(self, primitive_structure: List[PrimitiveAtomTemplate],
                               b_labels: Union[None, List[float], np.ndarray] = None):

        pdb_str = ""

        last_resi_id = None
        resi_idx = 0
        for primitive_atom_idx, pra_template in enumerate(primitive_structure):

            resi_id = pra_template.atom_source.source_residue
            resi_name = pra_template.atom_source.source_residue_name
            b_factor = 1. if b_labels is None else b_labels[primitive_atom_idx]
            coords = pra_template.coordinates

            if resi_id != last_resi_id:
                resi_idx += 1
                last_resi_id = resi_id

            atom_name = chr(65 + self.all_primitive_types.index(pra_template.primitive_type))

            pdb_str += f"ATOM  "  # 1-6
            pdb_str += f"{primitive_atom_idx + 1: >5} "  # Atom idx + space, 7-12
            pdb_str += f"{atom_name: >4}"  # Atom name, 13-16
            pdb_str += f" "  # Alternate location, 17
            pdb_str += f"{resi_name} "  # Residue name + space, 21
            pdb_str += f"{resi_id[1]}"  # Chain ID, 22
            pdb_str += f"{resi_idx: >4}"  # Residue sequence number, 23-26
            pdb_str += f"    "  # Insertion code + 3 spaces, 27-30
            pdb_str += f"{coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}"  # x, y, z coordinates, 31-54
            pdb_str += f"{1.:6.2f}"  # Occupancy, 55-60
            pdb_str += f"{b_factor:6.2f}          "  # Temp factor + 10 spaces, 61-76
            pdb_str += f"Pr"  # Element symbol, 77-78
            pdb_str += f"  "  # Charge, 79-80
            pdb_str += f"\n"

        return pdb_str
