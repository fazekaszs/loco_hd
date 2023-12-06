import codecs
import tarfile
import os
import warnings
import pickle

from pathlib import Path
from typing import Dict

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Entity import Entity
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom

# Set the necessary constants. The available predictor keys are the following:
# AF2: TS427, BAKER: TS473, BAKER-experimental: TS403, FEIG-R2: TS480, Zhang: TS129
PREDICTOR_KEY = "TS427"
TARFILE_ROOT = Path("../../data_sources/casp14")
TARGET_DIR = Path("../../workdir/casp14")


class InLinePDBParser(PDBParser):

    def from_str(self, pdb_id: str, pdb_content: str) -> Structure:

        with warnings.catch_warnings():
            if self.QUIET:
                warnings.filterwarnings("ignore", category=PDBConstructionWarning)

            self.header = None
            self.trailer = None
            self.structure_builder.init_structure(pdb_id)

            self._parse([line + "\n" for line in pdb_content.split("\n")])

            self.structure_builder.set_header(self.header)

            structure = self.structure_builder.get_structure()

        return structure


def filter_atoms(ref_structure: Entity, structure: Entity):

    for child_key in list(structure.child_dict.keys()):
        if child_key not in ref_structure.child_dict:
            structure.detach_child(child_key)
        elif type(structure.child_dict[child_key]) is Atom:
            filter_atoms(ref_structure.child_dict[child_key], structure.child_dict[child_key])


def main():

    # Extract the file names.
    tarfile_names = list(filter(lambda x: x.endswith(".tar.gz"), os.listdir(TARFILE_ROOT)))

    true_structures_tar = list(filter(lambda x: "casp14" in x, tarfile_names))
    [tarfile_names.remove(file_name) for file_name in true_structures_tar]

    # Read all the true structures.
    # First key in all_structures is the target structure's name (like TS1030),
    #  second key is either "true" (referring to the experimental structure) or
    #  a str number ("1"..."5", referring to the predicted structure).
    all_structures: Dict[str, Dict[str, Structure]] = dict()
    file_name: str
    for file_name in true_structures_tar:
        with tarfile.open(TARFILE_ROOT / file_name) as tf:
            tf_members = tf.getmembers()
            for member in tf_members:
                f = tf.extractfile(member)
                if f is not None:
                    content = codecs.getreader("utf-8")(f).read()
                    content = InLinePDBParser(QUIET=True).from_str("", content)

                    member_name = member.name.replace(".pdb", "")

                    all_structures[member_name] = {"true": content, }

                    print(f"True member name: {member_name},", end=" ")
                    print(f"# of atoms: {len(list(content.get_atoms()))},", end=" ")
                    print(f"# of structures so far: {len(all_structures)}")

                    f.close()

    # Read the predictor predictions.
    file_name: str
    for file_name in tarfile_names:
        with tarfile.open(TARFILE_ROOT / file_name) as tf:
            tf_members = tf.getmembers()
            tf_members = list(filter(lambda m: PREDICTOR_KEY in m.name, tf_members))
            for member in tf_members:

                member_name = member.name.split("/")[1]
                member_name = member_name.replace(PREDICTOR_KEY, "")
                member_name, member_id = member_name.split("_")

                if member_name not in all_structures:
                    continue

                try:
                    f = tf.extractfile(member)
                except KeyError as ke:
                    print(f"Member {member_name} failed with KeyError: {ke}! Skipping...")
                    continue

                content = codecs.getreader("utf-8")(f).read()
                content = InLinePDBParser(QUIET=True).from_str("", content)
                filter_atoms(all_structures[member_name]["true"], content)

                n_of_atoms = len(list(content.get_atoms()))
                n_of_atoms_true = len(list(all_structures[member_name]["true"].get_atoms()))

                if n_of_atoms != n_of_atoms_true:
                    f.close()
                    continue

                all_structures[member_name][member_id] = content

                print(f"{PREDICTOR_KEY} member name: {member_name},", end=" ")
                print(f"member id: {member_id},", end=" ")
                print(f"# of atoms: {n_of_atoms} (vs {n_of_atoms_true}),", end=" ")
                print(f"# of structures in member: {len(all_structures[member_name])}")

                f.close()

    all_structures = {key: value for key, value in all_structures.items() if len(value) > 1}

    predictor_target_dir = TARGET_DIR / f"{PREDICTOR_KEY}_results"
    if not os.path.exists(predictor_target_dir):
        os.mkdir(predictor_target_dir)

    predictor_target_file = predictor_target_dir / f"{PREDICTOR_KEY}_biopython_structures.pickle"
    with open(predictor_target_file, "wb") as f:
        pickle.dump(all_structures, f)


if __name__ == "__main__":
    main()
