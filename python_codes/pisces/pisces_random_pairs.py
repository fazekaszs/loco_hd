import json
import shutil
import math
import os
import random
import pickle
import numpy as np
import datetime

from pathlib import Path
from typing import List, Tuple
from time import time
from scipy import stats

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Model import Model

from loco_hd import LoCoHD, PrimitiveAtom, WeightFunction, PrimitiveAssigner, PrimitiveAtomTemplate, TagPairingRule

CURRENT_TIME = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
OUTPUT_FILENAME = "locohd_data.pisces"
WORKDIR_TARGET = Path("../workdir/pisces")
ASSIGNER_CONFIG_PATH = Path("../primitive_typings/coarse_grained_with_centroid.config.json")
PISCES_DIR_PATH = Path("../../../PycharmProjects/databases/pisces_220222")
RANDOM_SEED = 1994
MAX_N_OF_ANCHORS = 1500
WEIGHT_FUNCTION = ("uniform", [3, 10, ])
TAG_PAIRING_RULE = TagPairingRule({"accept_same": False})
UPPER_CUTOFF = 10


def is_anchor_atom(pra_template: PrimitiveAtomTemplate) -> bool:

    # return len(pra_template.atom_source.source_atom) == 0
    return pra_template.primitive_type == "Cent"


def get_anchors_and_primitive_atoms(pra_templates: List[PrimitiveAtomTemplate],
                                    model: Model) -> Tuple[List[int], List[PrimitiveAtom]]:

    anchors, primitive_atoms = list(), list()
    for idx, pra_template in enumerate(pra_templates):

        resi_id = pra_template.atom_source.source_residue
        resi_name = model[resi_id[2]][resi_id[3]].resname
        pra_source = f"{resi_id[2]}/{resi_id[3][1]}-{resi_name}"
        current_pra = PrimitiveAtom(pra_template.primitive_type, pra_source, pra_template.coordinates)
        primitive_atoms.append(current_pra)

        if is_anchor_atom(pra_template):
            anchors.append(idx)

    return anchors, primitive_atoms


def main():

    # Create working directory
    workdir_path = WORKDIR_TARGET / f"run_{CURRENT_TIME}"
    if os.path.exists(workdir_path):
        raise Exception(f"Workdir at {workdir_path} already exists!")
    else:
        os.mkdir(workdir_path)

    # Save parameters to the working directory
    with open(workdir_path / "params.json", "w") as f:
        json.dump({
            "assigner_config_path": str(ASSIGNER_CONFIG_PATH), "pisces_path": str(PISCES_DIR_PATH),
            "random_seed": RANDOM_SEED, "weight_function": WEIGHT_FUNCTION,
            "tag_pairing_rule": TAG_PAIRING_RULE.get_dbg_str(), "upper_cutoff": UPPER_CUTOFF
        }, f)

    # Save the assigner config to the working directory
    shutil.copyfile(ASSIGNER_CONFIG_PATH, workdir_path / "assigner_config.json")

    # Initialize the assigner and locohd
    primitive_assigner = PrimitiveAssigner(ASSIGNER_CONFIG_PATH)
    weight_function = WeightFunction(*WEIGHT_FUNCTION)
    lchd = LoCoHD(primitive_assigner.all_primitive_types, weight_function, TAG_PAIRING_RULE)

    # Collect the PDB file names
    pdb_files: List[str] = os.listdir(PISCES_DIR_PATH)
    pdb_files = list(filter(lambda x: x.endswith(".pdb"), pdb_files))
    random.seed(RANDOM_SEED)
    random.shuffle(pdb_files)

    time_per_anchor_list = list()
    len_list = list()
    for pdb_idx in range(len(pdb_files) - 1):

        time_start = time()

        path1 = str(PISCES_DIR_PATH / pdb_files[pdb_idx])
        path2 = str(PISCES_DIR_PATH / pdb_files[pdb_idx + 1])

        print(f"Starting {pdb_files[pdb_idx]} and {pdb_files[pdb_idx + 1]} - ", end="")

        # Load pdb files, assign primitive structures, and get the anchors and primitive atoms.
        protein1: Model = PDBParser(QUIET=True).get_structure("", path1)[0]
        protein2: Model = PDBParser(QUIET=True).get_structure("", path2)[0]

        pra_templates1 = primitive_assigner.assign_primitive_structure(protein1)
        pra_templates2 = primitive_assigner.assign_primitive_structure(protein2)

        anchors1, primitive_atoms1 = get_anchors_and_primitive_atoms(pra_templates1, protein1)
        anchors2, primitive_atoms2 = get_anchors_and_primitive_atoms(pra_templates2, protein2)

        # Pair anchor indices randomly together.
        if len(anchors1) < len(anchors2):
            anchors2 = random.sample(anchors2, len(anchors1))
        else:
            anchors1 = random.sample(anchors1, len(anchors2))

        anchor_pairs = [(x, y) for x, y in zip(anchors1, anchors2)]

        # Chop down the list of anchor indices if it exceeds an upper limit.
        anchor_pairs = anchor_pairs[:MAX_N_OF_ANCHORS] if len(anchor_pairs) > MAX_N_OF_ANCHORS else anchor_pairs

        print(f"Number of anchors: {len(anchor_pairs)} - ", end="")

        # Start LoCoHD calculations.
        lchd_scores = lchd.from_primitives(primitive_atoms1, primitive_atoms2, anchor_pairs, UPPER_CUTOFF)

        print(f"Calculation #{pdb_idx + 1} OK! Avg. LoCoHD: {np.mean(lchd_scores):.2%}")

        pdb_id1 = pdb_files[pdb_idx].replace(".pdb", "")
        pdb_id2 = pdb_files[pdb_idx + 1].replace(".pdb", "")

        cumulative_results = list()
        for anchor, lchd_score in zip(anchor_pairs, lchd_scores):
            pair_id1 = f"{pdb_id1}/{primitive_atoms1[anchor[0]].tag}"
            pair_id2 = f"{pdb_id2}/{primitive_atoms2[anchor[1]].tag}"
            cumulative_results.append((pair_id1, pair_id2, lchd_score))

        if os.path.exists(workdir_path / OUTPUT_FILENAME):
            with open(workdir_path / OUTPUT_FILENAME, "rb") as f:
                cumulative_results += pickle.load(f)

        with open(workdir_path / OUTPUT_FILENAME, "wb") as f:
            pickle.dump(cumulative_results, f)

        print("Cumulative results successfully saved!")

        time_end = time()

        # Time statistics
        len_list.append(len(anchor_pairs))
        time_per_anchor_list.append((time_end - time_start) / len(anchor_pairs))

        if len(len_list) > 2:

            len_mean = np.mean(len_list)
            len_std = np.std(len_list)

            time_per_anchor_mean = np.mean(time_per_anchor_list)
            time_per_anchor_std = np.std(time_per_anchor_list)

            remainers = (len(pdb_files) - (pdb_idx + 1))

            eta_mean = remainers * len_mean * time_per_anchor_mean
            eta_std = math.sqrt(
                (len_std * time_per_anchor_mean) ** 2 +
                (time_per_anchor_std * len_mean) ** 2
            ) * remainers

            eta_conf_int = stats.t.ppf(0.95, len(len_list) - 2) * eta_std / math.sqrt(len(len_list))

            completed_percent = (pdb_idx + 1) / len(pdb_files)

            print(f"ETA: {eta_mean:.1f} s +/- {eta_conf_int:.1f} s, Completed: {completed_percent:.2%}")


if __name__ == "__main__":
    main()
