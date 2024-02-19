import codecs
import sys
import tarfile
import os
import pickle
import re

from typing import Dict, List, Union

from Bio.PDB.Chain import Chain
from Bio.PDB.Entity import Entity
from Bio.PDB.Residue import Residue, DisorderedResidue
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.PDBIO import PDBIO

from config import (
    PREDICTOR_KEY,
    FILTER_TYPE,
    PRED_TAR_DIR,
    REFS_TAR_DIR,
    EXTRACTOR_OUTPUT_DIR
)
from utils import InLinePDBParser, PDBIOFakeFileHandler

# Do only a test run and don't save anything.
TEST_RUN = False


def remove_disordered_elements(structure: Structure):

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


def read_in_ref_structures():

    # Extract the file names.
    refs_tarfile_names = list(filter(lambda x: x.endswith(".tar.gz"), os.listdir(REFS_TAR_DIR)))
    print(f"Found {len(refs_tarfile_names)} reference structure tarfiles!")

    # Initialize necessary directories for the reference structures.
    all_structures: Dict[str, Dict[str, Structure]] = dict()

    # Read all the true structures.
    file_name: str
    for file_name in refs_tarfile_names:

        tf = tarfile.open(REFS_TAR_DIR / file_name)
        tf_members = tf.getmembers()

        print(f"[ {file_name} ] I see {len(tf_members)} members in this file!")

        for member in tf_members:

            f = tf.extractfile(member)

            if f is None:
                print(f"[ {file_name}/{member.name} ] This was not a file! Continuing...")
                continue

            content = codecs.getreader("utf-8")(f).read()
            content = InLinePDBParser(QUIET=True).from_str("", content)

            print(f"[ {file_name}/{member.name} ] parsed!")

            member_name = member.name.replace(".pdb", "")

            # This check is for CASP15 file structure compatibility:
            if member_name.startswith("./"):
                member_name = member_name[2:]

            all_structures[member_name] = {"true": content, }

            # Print status.
            n_of_atoms = len(list(content.get_atoms()))
            n_of_ref_structures = len(all_structures)

            print(
                f"[ {file_name}/{member.name} ] has {n_of_atoms} atoms. "
                f"Number of reference structures so far: {n_of_ref_structures}."
            )

            f.close()

        tf.close()

    # Print reference read status.
    n_of_ref_structures = len(all_structures)
    print(f"Reference structure collection finished! Number of reference structures: {n_of_ref_structures}.")

    return all_structures


def read_in_pred_structures(all_structures: Dict[str, Dict[str, Structure]]):

    # Extract the file names.
    pred_tarfile_names = list(filter(lambda x: x.endswith(".tar.gz"), os.listdir(PRED_TAR_DIR)))
    print(f"Found {len(pred_tarfile_names)} predicted structure tarfiles!")

    # Filter filenames according to the reference filenames.
    valid_structure_names = set(all_structures.keys()).intersection({
        name.replace(".tar.gz", "") for name in pred_tarfile_names
    })
    print(f"{len(valid_structure_names)} predicted structures are present in the reference structure database!")

    # Delete unnecessary reference structures.
    all_structures = {
        structure_name: structure
        for structure_name, structure in all_structures.items()
        if structure_name in valid_structure_names
    }
    print(f"Unnecessary structures were thrown away!")

    # Create prediction name checker regex.
    pred_name_checker = re.compile(r"^([T|H][0-9]{4}(?:[s|v][0-9])?)(TS[0-9]{3})_([0-9])o?$")

    # Read the predictor predictions.
    for structure_name in valid_structure_names:

        file_name = structure_name + ".tar.gz"

        tf = tarfile.open(PRED_TAR_DIR / file_name)

        tf_members = tf.getmembers()
        tf_members = list(filter(lambda m: PREDICTOR_KEY in m.name, tf_members))

        print(f"[ {file_name} ] I see {len(tf_members)} members in this file!")

        for member in tf_members:

            prediction_full_str = member.name.split("/")[-1]

            match_obj = re.fullmatch(pred_name_checker, prediction_full_str)
            if match_obj is None:
                sys.exit(f"ERROR: {prediction_full_str} does not match the expected pattern!")

            member_id = match_obj.groups()[2]

            try:
                f = tf.extractfile(member)
            except KeyError as ke:
                print(f"Member {structure_name} failed with KeyError: {ke}! Skipping...")
                continue

            content = codecs.getreader("utf-8")(f).read()
            content = InLinePDBParser(QUIET=True).from_str("", content)

            print(f"{PREDICTOR_KEY} member name: {structure_name},", end=" ")
            print(f"member id: {member_id},", end=" ")
            print(f"# of structures in member: {len(all_structures[structure_name])}")

            all_structures[structure_name][member_id] = content

            f.close()

        tf.close()

    # Keep structure bundles only where there are predicted structures too.
    for structure_name in list(all_structures.keys()):

        n_of_structures = len(all_structures[structure_name])
        print(f"{structure_name} bundle contains {n_of_structures} structures!")
        if n_of_structures == 1:  # only "true" is present
            print(f"Deleting bundle {structure_name}...")
            del all_structures[structure_name]

    return all_structures


def normalize_ref_structures(all_structures: Dict[str, Dict[str, Structure]]):

    for structure_name, structure_bundle in all_structures.items():

        ref_structure = structure_bundle["true"]
        remove_disordered_elements(ref_structure)

        print(f"Reference structure {structure_name} normalized!")


def apply_filtering(all_structures: Dict[str, Dict[str, Structure]]):

    # Define helper functions.
    def loose_atom_filter(ref_structure: Entity, mod_structure: Entity):

        for child_key in list(mod_structure.child_dict.keys()):
            if child_key not in ref_structure.child_dict:
                mod_structure.detach_child(child_key)
            elif type(mod_structure.child_dict[child_key]) is not Atom:
                loose_atom_filter(ref_structure.child_dict[child_key], mod_structure.child_dict[child_key])

    def strict_atom_filter(structures: List[Entity]):

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
            strict_atom_filter(common_children)

    # Do the filtration.
    for structure_name, structure_bundle in all_structures.items():

        if FILTER_TYPE == "loose":

            for structure_id, structure in structure_bundle.items():
                if structure_id == "true":
                    continue
                loose_atom_filter(structure_bundle["true"], structure)

        elif FILTER_TYPE == "strict":

            bundle_list = list(structure_bundle.values())
            strict_atom_filter(bundle_list)

    # Report.
    print(f"Atom counts after filtration:")
    for structure_name, structure_bundle in all_structures.items():

        out_str = f"{structure_name}: "
        for structure_id, structure in structure_bundle.items():
            n_of_atoms = len(list(structure.get_atoms()))
            out_str += f"({structure_id}, {n_of_atoms}) "
        print(out_str)


def add_chain_names(all_structures: Dict[str, Dict[str, Structure]]):

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


def main():

    all_structures = read_in_ref_structures()
    all_structures = read_in_pred_structures(all_structures)
    normalize_ref_structures(all_structures)  # in-place
    apply_filtering(all_structures)  # in-place
    add_chain_names(all_structures)  # in-place
    all_structures = stringify_structures(all_structures)

    print(f"Number of structures to be saved: {len(all_structures)}")

    if TEST_RUN:
        print("It was only a test run! No saving occurred!")
        sys.exit(0)

    predictor_target_dir = EXTRACTOR_OUTPUT_DIR
    if not os.path.exists(predictor_target_dir):
        os.mkdir(predictor_target_dir)

    print(f"Target directory is: {predictor_target_dir}")

    predictor_target_file_name = f"filtered_structures.pickle"
    predictor_target_file = predictor_target_dir / predictor_target_file_name
    with open(predictor_target_file, "wb") as f:
        pickle.dump(all_structures, f)

    print(f"Target file saved as: {predictor_target_file_name}")


if __name__ == "__main__":
    main()
