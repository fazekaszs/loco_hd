import numpy as np
import json

from pathlib import Path
from typing import Union, List, Tuple, Dict
from dataclasses import dataclass

from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue

AtomHolders = Union[Structure, Model, Chain]
ResiFullIdType = Tuple[str, int, str, Tuple[str, int, str]]  # structure id, model id, chain id, residue id


@dataclass
class PrimitiveAtomSource:
    """
        Contains the residue source for the primitive atom, as well as the atomic-level details.
        The source_atom field contains info about how the primitive atom was constructed; it is
        either an empty list (denoting full-residue centroid), a list with only one element
        (denoting one source atom name), or a list with multiple elements (denoting that the
        primitive atom is a centroid of several atoms).
    """

    source_residue: ResiFullIdType
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


class PrimitiveAssigner:
    """
        Aims to contain the primitive typing scheme and is able to convert protein structures (i.e.,
        PDB files) to a list of PrimitiveAtomTemplates (see the assign_primitive_strucutre method).
    """

    def __init__(self, config_path: Path):

        with open(config_path, "r") as f:
            config: Dict[str, List[str]] = json.load(f)

        self.converter_dict = {
            "*:*": list(),  # A list of primitive types (str)
            "*:A-B": list(),  # A list of 2-tuples: [(N-tuple specifying atoms, primitive type), ... ]
            "*:A": list(),  # A list of 2-tuples: [(atom name, primitive type), ...]
            "X:*": dict(),  # A dict with residue names (str) as keys and a list of prim. types (like *:*) as a value
            "X:A-B": dict(),  # A dict with residue names (str) as keys and a list of 2-tuples (like *:A-B) as a value
            "X:A": dict(),  # A dict with residue names (str) as keys and a list of 2-tuples (like *:A) as a value
        }

        for primitive_type, atom_name_list in config.items():

            for atom_name_list_element in atom_name_list:

                resi_specifier, atom_specifier = atom_name_list_element.split(":")

                is_universal_resi = resi_specifier == "*"
                is_universal_atom = atom_specifier == "*"
                is_centroid_atom = "-" in atom_specifier

                if is_universal_resi and is_universal_atom:
                    self.converter_dict["*:*"].append(primitive_type)

                elif is_universal_resi and is_centroid_atom:
                    centroid_atom_list = tuple(atom_specifier.split("-"))
                    self.converter_dict["*:A-B"].append((centroid_atom_list, primitive_type))

                elif is_universal_resi:
                    self.converter_dict["*:A"].append((atom_specifier, primitive_type))

                elif is_universal_atom:
                    if resi_specifier in self.converter_dict["X:*"]:
                        self.converter_dict["X:*"][resi_specifier].append(primitive_type)
                    else:
                        self.converter_dict["X:*"][resi_specifier] = [primitive_type, ]

                elif is_centroid_atom:
                    centroid_atom_list = tuple(atom_specifier.split("-"))
                    if resi_specifier in self.converter_dict["X:A-B"]:
                        self.converter_dict["X:A-B"][resi_specifier].append((centroid_atom_list, primitive_type))
                    else:
                        self.converter_dict["X:A-B"][resi_specifier] = [(centroid_atom_list, primitive_type), ]

                else:
                    if resi_specifier in self.converter_dict["X:A"]:
                        self.converter_dict["X:A"][resi_specifier].append((atom_specifier, primitive_type))
                    else:
                        self.converter_dict["X:A"][resi_specifier] = [(atom_specifier, primitive_type), ]

    @property
    def all_primitive_types(self) -> List[str]:

        primitive_types = set()

        for pr_type in self.converter_dict["*:*"]:
            primitive_types.add(pr_type)

        for _, pr_type in self.converter_dict["*:A-B"]:
            primitive_types.add(pr_type)

        for _, pr_type in self.converter_dict["*:A"]:
            primitive_types.add(pr_type)

        for resi_name in self.converter_dict["X:*"]:
            for pr_type in self.converter_dict["X:*"][resi_name]:
                primitive_types.add(pr_type)

        for resi_name in self.converter_dict["X:A-B"]:
            for _, pr_type in self.converter_dict["X:A-B"][resi_name]:
                primitive_types.add(pr_type)

        for resi_name in self.converter_dict["X:A"]:
            for _, pr_type in self.converter_dict["X:A"][resi_name]:
                primitive_types.add(pr_type)

        return list(primitive_types)

    @all_primitive_types.setter
    def all_primitive_types(self, value):
        raise Exception("Cannot set all_primitive_types directly, since it depends on the config file!")

    def assign_primitive_structure(self, structure: AtomHolders) -> List[PrimitiveAtomTemplate]:

        out = list()

        resi: Residue
        for resi in structure.get_residues():

            resi_id = resi.full_id

            # CATEGORY *:*
            full_centroid_coord = None
            if len(self.converter_dict["*:*"]) > 0:

                full_centroid_coord = resi.center_of_mass(geometric=True)
                for primitive_type in self.converter_dict["*:*"]:

                    pra_source = PrimitiveAtomSource(resi_id, list())
                    pra_template = PrimitiveAtomTemplate(primitive_type, full_centroid_coord, pra_source)
                    out.append(pra_template)

            # CATEGORY *:A-B
            for atom_names, primitive_type in self.converter_dict["*:A-B"]:

                partial_centroid_coord = list()
                for atom_name in atom_names:

                    if atom_name in resi:
                        partial_centroid_coord.append(resi[atom_name].coord)
                    else:
                        print(f"Warning: no atom named {atom_name} in residue {resi_id} ({resi.resname})!")
                        break
                else:

                    partial_centroid_coord = np.mean(partial_centroid_coord, axis=0)

                    pra_source = PrimitiveAtomSource(resi_id, atom_names)
                    pra_template = PrimitiveAtomTemplate(primitive_type, partial_centroid_coord, pra_source)
                    out.append(pra_template)

            # CATEGORY *:A
            for atom_name, primitive_type in self.converter_dict["*:A"]:

                if atom_name not in resi:
                    print(f"Warning: no atom named {atom_name} in residue {resi_id} ({resi.resname})!")
                    continue

                pra_source = PrimitiveAtomSource(resi_id, [atom_name, ])
                pra_template = PrimitiveAtomTemplate(primitive_type, resi[atom_name].coord, pra_source)
                out.append(pra_template)

            # CATEGORY X:*
            if resi.resname in self.converter_dict["X:*"]:

                if full_centroid_coord is None:
                    full_centroid_coord = resi.center_of_mass(geometric=True)

                for primitive_type in self.converter_dict["X:*"][resi.resname]:

                    pra_source = PrimitiveAtomSource(resi_id, list())
                    pra_template = PrimitiveAtomTemplate(primitive_type, full_centroid_coord, pra_source)
                    out.append(pra_template)

            # CATEGORY X:A-B
            if resi.resname in self.converter_dict["X:A-B"]:

                for atom_names, primitive_type in self.converter_dict["X:A-B"][resi.resname]:

                    partial_centroid_coord = list()
                    for atom_name in atom_names:

                        if atom_name in resi:
                            partial_centroid_coord.append(resi[atom_name].coord)
                        else:
                            print(f"Warning: no atom named {atom_name} in residue {resi_id} ({resi.resname})!")
                            break
                    else:

                        partial_centroid_coord = np.mean(partial_centroid_coord, axis=0)

                        pra_source = PrimitiveAtomSource(resi_id, atom_names)
                        pra_template = PrimitiveAtomTemplate(primitive_type, partial_centroid_coord, pra_source)
                        out.append(pra_template)

            # CATEGORY X:A
            if resi.resname in self.converter_dict["X:A"]:

                for atom_name, primitive_type in self.converter_dict["X:A"][resi.resname]:

                    if atom_name not in resi:
                        print(f"Warning: no atom named {atom_name} in residue {resi_id} ({resi.resname})!")
                        continue

                    pra_source = PrimitiveAtomSource(resi_id, [atom_name, ])
                    pra_template = PrimitiveAtomTemplate(primitive_type, resi[atom_name].coord, pra_source)
                    out.append(pra_template)

        return out

    def generate_primitive_pdb(self, structure: AtomHolders):

        primitive_structure = self.assign_primitive_structure(structure)
        pdb_str = ""

        last_resi_id = None
        resi_idx = 0
        for primitive_atom_idx, pra_template in enumerate(primitive_structure):

            resi_id = pra_template.atom_source.source_residue
            coords = pra_template.coordinates

            if resi_id != last_resi_id:
                resi_idx += 1
                last_resi_id = resi_id

            atom_name = chr(65 + self.all_primitive_types.index(pra_template.primitive_type))

            if type(structure) == Structure:
                resi_name = structure[resi_id[1]][resi_id[2]][resi_id[3]].resname
            elif type(structure) == Model:
                resi_name = structure[resi_id[2]][resi_id[3]].resname
            else:
                resi_name = structure[resi_id[3]].resname

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
            pdb_str += f"{1.:6.2f}          "  # Temp factor + 10 spaces, 61-76
            pdb_str += f"Pr"  # Element symbol, 77-78
            pdb_str += f"  "  # Charge, 79-80
            pdb_str += f"\n"

        return pdb_str
