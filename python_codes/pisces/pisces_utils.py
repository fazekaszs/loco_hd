import os
import subprocess as subp
import sys
import pickle

from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np

RING_FILE_PATH = Path("../../ring-3.0.0/ring/bin/ring")
PISCES_DIR_PATH = Path("../../../../PycharmProjects/databases/pisces_241209")

OUTPUT_PATH = Path("../../workdir/pisces/ring_result")
PISCES_LOCOHD_FILE_PATH = Path("../../workdir/pisces/run_2023-02-08-12-50-23/locohd_data.pisces")

INTERACTIONS = [
    "HBOND", "VDW", "SSBOND", "IONIC", "PIPISTACK", "PICATION"
]
RESI_TLCS = [
    "GLY", "ALA", "VAL", "ILE", "LEU", "PHE",
    "SER", "THR", "TYR", "ASP", "GLU", "ASN",
    "GLN", "TRP", "HIS", "MET", "PRO", "CYS",
    "ARG", "LYS"
]


class EnvironmentPairList:

    def __init__(self, ring_data: Dict[str, np.ndarray], locohd_data: List[Tuple[str, str, float]]):

        self.training_data = None  # cached training data

        self.side1_ids = list()
        self.side2_ids = list()
        self.side1_interactions = list()
        self.side2_interactions = list()
        self.locohd_values = list()

        for side1_id, side2_id, locohd_score in locohd_data:

            side1_interaction = ring_data.get(side1_id, np.zeros(len(INTERACTIONS)))
            side2_interaction = ring_data.get(side2_id, np.zeros(len(INTERACTIONS)))

            self.side1_ids.append(side1_id)
            self.side2_ids.append(side2_id)
            self.side1_interactions.append(side1_interaction)
            self.side2_interactions.append(side2_interaction)
            self.locohd_values.append(locohd_score)

    def __len__(self):
        return len(self.locohd_values)


def run_ring() -> None:
    """
    Runs RING on every pdb file inside a (PISCES) dictionary.
    """

    # Collecting input pdb file names
    pdb_file_names = os.listdir(PISCES_DIR_PATH)
    pdb_file_names = list(filter(lambda x: x.endswith(".pdb"), pdb_file_names))

    # Creating RING input file
    ring_input_file_name = "ring_pdb_list.txt"
    ring_input_str = "\n".join([
        str((PISCES_DIR_PATH / pdb_file_name).resolve())
        for pdb_file_name in pdb_file_names
    ])

    with open(OUTPUT_PATH / ring_input_file_name, "w") as f:
        f.write(ring_input_str)

    # Running RING analysis with the input file
    to_subp_run = [
        str(RING_FILE_PATH),
        "-I", str(OUTPUT_PATH / ring_input_file_name),
        "--out_dir", str(OUTPUT_PATH)
    ]

    exit_code = subp.Popen(to_subp_run).wait()

    if exit_code != 0:
        sys.exit(f"Nonzero exitcode encountered in RING: {exit_code}!")

    # Cleanup
    ring_node_files = os.listdir(OUTPUT_PATH)
    ring_node_files = list(filter(lambda x: x.endswith("_ringNodes"), ring_node_files))
    for file_name in ring_node_files:
        os.system(f"rm {str(OUTPUT_PATH.resolve() / file_name)}")


def process_ring_result() -> Dict[str, np.ndarray]:
    """
    Read the ringEdges files and collect the data from them. The output is a dictionary with residue
    specifying keys (pdb_id/chain_id/resi_id-resi_type) and interaction count vectors.

    :return: The residue contact count dictionary.
    """

    ring_edge_file_names = os.listdir(OUTPUT_PATH)
    ring_edge_file_names = list(filter(lambda x: x.endswith(".pdb_ringEdges"), ring_edge_file_names))

    out = dict()
    for file_name in ring_edge_file_names:

        pdb_code = file_name.split(".")[0]

        with open(OUTPUT_PATH / file_name, "r") as f:
            data = f.read()

        data = data.split("\n")[1:]
        data = list(filter(lambda x: len(x) != 0, data))

        for line in data:

            line = line.split("\t")

            side1 = line[0].split(":")
            side1 = f"{pdb_code}/{side1[0]}/{side1[1]}-{side1[3]}"

            side2 = line[2].split(":")
            side2 = f"{pdb_code}/{side2[0]}/{side2[1]}-{side2[3]}"

            interaction = line[1].split(":")[0]

            counts1 = out.get(side1, np.zeros(len(INTERACTIONS)))
            counts1[INTERACTIONS.index(interaction)] += 1
            out[side1] = counts1

            counts2 = out.get(side2, np.zeros(len(INTERACTIONS)))
            counts2[INTERACTIONS.index(interaction)] += 1
            out[side2] = counts2

        print("\rNumber of residues:", len(out), end="")

    # Cleanup
    ring_edge_files = os.listdir(OUTPUT_PATH)
    ring_edge_files = list(filter(lambda x: x.endswith("_ringEdges"), ring_edge_files))
    for file_name in ring_edge_files:
        os.system(f"rm {str(OUTPUT_PATH.resolve() / file_name)}")

    return out


def get_ring_data() -> Dict[str, np.ndarray]:
    """
    Calls the run_ring and process_ring_result functions if necessary, or loads the corresponding, already
    saved ring datafile. Returns a dictionary with residue ids as keys and interaction count vectors as
    values. For the key formats see the process_ring_result function documentation.

    :return: The residue interaction count dictionary.
    """

    ring_out_filename = "collected.pickle"

    if os.path.exists(OUTPUT_PATH / ring_out_filename):

        print(f"{str(OUTPUT_PATH / ring_out_filename)} already exists! Using this file...")
        with open(OUTPUT_PATH / ring_out_filename, "rb") as f:
            ring_data: Dict[str, np.ndarray] = pickle.load(f)

    else:

        run_ring()
        ring_data = process_ring_result()
        with open(OUTPUT_PATH / ring_out_filename, "wb") as f:
            pickle.dump(ring_data, f)

    return ring_data


def tlc_to_one_hot(tlc: str):
    """
    Converts the three-letter code of an amino acid to a one-hot encoding.

    :param tlc: The three-letter code of the amino acid.
    :return: The one-hot encoded vector.
    """

    out = np.zeros(len(RESI_TLCS))
    out[RESI_TLCS.index(tlc)] = 1

    return out


def merge_datasets(ring_data: Dict[str, np.ndarray]) -> EnvironmentPairList:
    """
    Merge the LoCoHD dataset (resi1, resi2, lchd score) and the RING dataset (resi keys, contact count vectors).
    Creates the full dataset for the training of a neural network.

    :param ring_data:
    :return:
    """

    with open(PISCES_LOCOHD_FILE_PATH, "rb") as f:
        locohd_data: List[Tuple[str, str, float]] = pickle.load(f)

    merged_data = EnvironmentPairList(ring_data, locohd_data)

    return merged_data