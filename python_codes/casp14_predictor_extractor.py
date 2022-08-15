import codecs
import tarfile
import os
import warnings
import pickle
from pathlib import Path
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Entity import Entity
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom


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
        elif type(structure.child_dict[child_key]) != Atom:
            filter_atoms(ref_structure.child_dict[child_key], structure.child_dict[child_key])


def main():

    # AF2: TS427
    # BAKER: TS473
    # BAKER-experimental: TS403
    # FEIG-R2: TS480
    # Zhang: TS129
    predictor_key = "TS129"

    # Extract the file names.
    tarfile_root = Path("/home/fazekaszs/CoreDir/PhD/PDB/casp14")
    tarfile_names = list(filter(lambda x: x.endswith(".tar.gz"), os.listdir(tarfile_root)))

    true_structures_tar = list(filter(lambda x: "casp14" in x, tarfile_names))
    [tarfile_names.remove(file_name) for file_name in true_structures_tar]

    # Read all the true structures.
    all_structures = dict()
    file_name: str
    for file_name in true_structures_tar:
        with tarfile.open(tarfile_root / file_name) as tf:
            tf_members = tf.getmembers()
            for member in tf_members:
                f = tf.extractfile(member)
                if f is not None:
                    content = codecs.getreader("utf-8")(f).read()
                    content = InLinePDBParser(QUIET=True).from_str("", content)

                    member_name = member.name.replace(".pdb", "")

                    all_structures[member_name] = [content, ]
                    print(f"True member name: {member_name}, "
                          f"# of atoms: {len(list(content.get_atoms()))}, "
                          f"# of structures so far: {len(all_structures)}")
                    f.close()

    # Read the predictor predictions.
    file_name: str
    for file_name in tarfile_names:
        with tarfile.open(tarfile_root / file_name) as tf:
            tf_members = tf.getmembers()
            tf_members = list(filter(lambda m: predictor_key in m.name, tf_members))
            for member in tf_members:

                member_name = member.name.split("/")[1]
                member_name = member_name.replace(predictor_key, "")
                member_name = member_name.split("_")[0]

                if member_name not in all_structures:
                    continue

                try:
                    f = tf.extractfile(member)
                except KeyError as ke:
                    print(f"Member {member_name} failed with KeyError: {ke}! Skipping...")
                    continue

                content = codecs.getreader("utf-8")(f).read()
                content = InLinePDBParser(QUIET=True).from_str("", content)
                filter_atoms(all_structures[member_name][0], content)

                n_of_atoms = len(list(content.get_atoms()))
                n_of_atoms_true = len(list(all_structures[member_name][0].get_atoms()))

                if n_of_atoms != n_of_atoms_true:
                    f.close()
                    continue

                all_structures[member_name].append(content)
                print(f"{predictor_key} member name: {member_name}, "
                      f"# of atoms: {n_of_atoms} (vs {n_of_atoms_true}), "
                      f"# of structures in member: {len(all_structures[member_name])}")
                f.close()

    all_structures = {key: value for key, value in all_structures.items() if len(value) > 1}

    if not os.path.exists(f"./workdir/{predictor_key}_results"):
        os.mkdir(f"./workdir/{predictor_key}_results")

    with open(f"./workdir/{predictor_key}_results/{predictor_key}_structures.pickle", "wb") as f:
        pickle.dump(all_structures, f)


if __name__ == "__main__":
    main()
