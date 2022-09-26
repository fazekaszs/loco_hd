import numpy as np

from pathlib import Path
from typing import Union, List, Tuple

from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom

from Bio.PDB.PDBParser import PDBParser


class PrimitiveAssigner:

    def __init__(self, config_path: Path):

        with open(config_path, "r") as f:
            config = f.read()

        config = config.replace(" ", "").replace("\n", "").split("[")[1:]

        self.converter_dict = {
            "*:*": list(),  # A list of primitive types (str)
            "*:A-B": list(),  # A list of 2-tuples: [(N-tuple specifying atoms, primitive type), ... ]
            "*:A": list(),  # A list of 2-tuples: [(atom name, primitive type), ...]
            "X:*": dict(),  # A dict with residue names (str) as keys and a list of 2-tuples (like *:*) as a value
            "X:A-B": dict(),  # A dict with residue names (str) as keys and a list of 2-tuples (like *:A-B) as a value
            "X:A": dict(),  # A dict with residue names (str) as keys and a list of 2-tuples (like *:A) as a value
        }

        for config_line in config:

            config_line = config_line.split("]")

            assert len(config_line) == 2, "Syntax error with \"[\", \"]\" type brackets!"

            primitive_type = config_line[0]

            for element in config_line[1].split(","):

                resi_specifier, atom_specifier = element.split(":")

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

    def assign_primitive_strucutre(self,
                                   structure: Union[Structure, Model, Chain]) -> List[Tuple[str, np.ndarray, str]]:

        out = list()

        resi: Residue
        for resi in structure.get_residues():

            resi_id = list(resi.full_id)
            resi_id[3] = resi_id[3][1]
            resi_id = "/".join(map(str, resi_id))  # structure_id/model_id/chain_id/resi_id

            # CATEGORY *:*
            full_centroid_coord = None
            if len(self.converter_dict["*:*"]) > 0:
                full_centroid_coord = resi.center_of_mass(geometric=True)
                for primitive_type in self.converter_dict["*:*"]:
                    out.append((primitive_type, full_centroid_coord, resi_id))

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
                    out.append((primitive_type, partial_centroid_coord, resi_id))

            # CATEGORY *:A
            for atom_name, primitive_type in self.converter_dict["*:A"]:
                if atom_name not in resi:
                    print(f"Warning: no atom named {atom_name} in residue {resi_id} ({resi.resname})!")
                    continue
                out.append((primitive_type, resi[atom_name].coord, resi_id))

            # CATEGORY X:*
            if resi.resname in self.converter_dict["X:*"]:
                if full_centroid_coord is None:
                    full_centroid_coord = resi.center_of_mass(geometric=True)
                for primitive_type in self.converter_dict["X:*"][resi.resname]:
                    out.append((primitive_type, full_centroid_coord, resi_id))

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
                        out.append((primitive_type, partial_centroid_coord, resi_id))

            # CATEGORY X:A
            if resi.resname in self.converter_dict["X:A"]:
                for atom_name, primitive_type in self.converter_dict["X:A"][resi.resname]:
                    if atom_name not in resi:
                        print(f"Warning: no atom named {atom_name} in residue {resi_id} ({resi.resname})!")
                        continue
                    out.append((primitive_type, resi[atom_name].coord, resi_id))

        return out

    def generate_primitive_pdb(self, structure: Union[Structure, Model, Chain]):

        primitive_structure = self.assign_primitive_strucutre(structure)
        pdb_str = ""

        last_resi_id = None
        resi_idx = 0
        primitive_types = list()
        for primitive_atom_idx, (primitive_type, coords, resi_id) in enumerate(primitive_structure):

            if resi_id != last_resi_id:
                resi_idx += 1
                last_resi_id = resi_id

            if primitive_type not in primitive_types:
                primitive_types.append(primitive_type)

            atom_name = chr(65 + primitive_types.index(primitive_type))

            resi_id = resi_id.split("/")[1:]
            resi_id[0] = int(resi_id[0])
            resi_id[2] = int(resi_id[2])

            if type(structure) == Structure:
                resi_name = structure[resi_id[0]][resi_id[1]][resi_id[2]].resname
            elif type(structure) == Model:
                resi_name = structure[resi_id[1]][resi_id[2]].resname
            else:
                resi_name = structure[resi_id[2]].resname

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


def main():

    config_path = Path("primitive_typings/atom_converter_config2")
    pa = PrimitiveAssigner(config_path)

    structure = PDBParser(QUIET=True).get_structure("test", "./workdir/pdb_files/h5/H5_288Kfit_1.pdb")
    out = pa.generate_primitive_pdb(structure)

    pass


if __name__ == "__main__":
    main()
