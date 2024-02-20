import os
import codecs
import tarfile
import re
import sys

from typing import Dict, Set
from pathlib import Path

from Bio.PDB.Structure import Structure

from .inline_pdb_parser import InLinePDBParser


def read_in_ref_structures(refs_tar_dir: Path, ignored_structures: Set[str]) -> Dict[str, Dict[str, Structure]]:

    # Extract the file names.
    refs_tarfile_names = list(filter(lambda x: x.endswith(".tar.gz"), os.listdir(refs_tar_dir)))
    print(f"Found {len(refs_tarfile_names)} reference structure tarfiles!")

    # Initialize necessary directories for the reference structures.
    all_structures: Dict[str, Dict[str, Structure]] = dict()

    # Read all the true structures.
    file_name: str
    for file_name in refs_tarfile_names:

        tf = tarfile.open(refs_tar_dir / file_name)
        tf_members = tf.getmembers()

        print(f"[ {file_name} ] I see {len(tf_members)} members in this file!")

        for member in tf_members:

            f = tf.extractfile(member)

            if f is None:
                print(f"[ {file_name}/{member.name} ] This was not a file! Continuing...")
                continue

            member_name = member.name.replace(".pdb", "")

            # This check is for CASP15 file structure compatibility:
            if member_name.startswith("./"):
                member_name = member_name[2:]

            # Check whether the structure should be ignored.
            if member_name in ignored_structures:
                print(f"[ {file_name}/{member.name} ] ignored!")
                continue

            # Read in the structure.
            content = codecs.getreader("utf-8")(f).read()
            content = InLinePDBParser(QUIET=True).from_str("", content)

            print(f"[ {file_name}/{member.name} ] parsed!")

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


def read_in_pred_structures(
        all_structures: Dict[str, Dict[str, Structure]],
        pred_tar_dir: Path,
        predictor_key: str
) -> Dict[str, Dict[str, Structure]]:

    # Extract the file names.
    pred_tarfile_names = list(filter(lambda x: x.endswith(".tar.gz"), os.listdir(pred_tar_dir)))
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

        tf = tarfile.open(pred_tar_dir / file_name)

        tf_members = tf.getmembers()
        tf_members = list(filter(lambda m: predictor_key in m.name, tf_members))

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

            print(f"{predictor_key} member name: {structure_name},", end=" ")
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

