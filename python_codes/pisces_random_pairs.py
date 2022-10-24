import math
import os
import random
import pickle
import numpy as np

from pathlib import Path
from typing import List
from time import time
from scipy import stats

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Model import Model

from loco_hd import LoCoHD, PrimitiveAtom
from atom_converter_utils import PrimitiveAssigner, PrimitiveAtomTemplate


def from_pra_template(pra_template: PrimitiveAtomTemplate, model: Model) -> PrimitiveAtom:

    resi_id = pra_template.atom_source.source_residue
    resi_name = model[resi_id[2]][resi_id[3]].resname
    pra_source = f"{resi_id[2]}/{resi_id[3][1]}-{resi_name}"
    return PrimitiveAtom(pra_template.primitive_type, pra_source, pra_template.coordinates)


def main():

    random.seed(1994)

    pisces_path = Path("/home/fazekaszs/PycharmProjects/databases/pisces_220222")
    save_path = Path("workdir/pisces")

    primitive_assigner = PrimitiveAssigner(Path("primitive_typings/all_atom_with_centroid.config.json"))

    # out_file_name = "results_uniform-3-10_only-hetero-contacts.pickle"
    # out_file_name = "results_uniform-3-10_all-contacts.pickle"
    out_file_name = "results_kumaraswamy-3-10-2-5_only-hetero-contacts.pickle"

    lchd = LoCoHD(primitive_assigner.all_primitive_types, ("kumaraswamy", [3, 10, 2, 5]))
    only_hetero_contacts = True
    upper_cutoff = 10

    pdb_files: List[str] = os.listdir(pisces_path)
    pdb_files = list(filter(lambda x: x.endswith(".pdb"), pdb_files))
    random.shuffle(pdb_files)

    time_per_anchor_list = list()
    len_list = list()
    for pdb_idx in range(len(pdb_files) - 1):

        time_start = time()

        path1 = str(pisces_path / pdb_files[pdb_idx])
        path2 = str(pisces_path / pdb_files[pdb_idx + 1])

        protein1: Model = PDBParser(QUIET=True).get_structure("", path1)[0]
        protein2: Model = PDBParser(QUIET=True).get_structure("", path2)[0]

        pra_templates1 = primitive_assigner.assign_primitive_structure(protein1)
        pra_templates2 = primitive_assigner.assign_primitive_structure(protein2)

        primitive_atoms1 = list(map(lambda x: from_pra_template(x, protein1), pra_templates1))
        primitive_atoms2 = list(map(lambda x: from_pra_template(x, protein2), pra_templates2))

        anchors1 = [idx for idx, pra_template in enumerate(pra_templates1) if pra_template.primitive_type == "Cent"]
        anchors2 = [idx for idx, pra_template in enumerate(pra_templates2) if pra_template.primitive_type == "Cent"]

        if len(anchors1) < len(anchors2):
            anchors2 = random.sample(anchors2, len(anchors1))
        else:
            anchors1 = random.sample(anchors1, len(anchors2))

        anchor_pairs = [(x, y) for x, y in zip(anchors1, anchors2)]

        lchd_scores = lchd.from_primitives(primitive_atoms1, primitive_atoms2,
                                           anchor_pairs, only_hetero_contacts, upper_cutoff)

        pdb_id1 = pdb_files[pdb_idx].replace(".pdb", "")
        pdb_id2 = pdb_files[pdb_idx + 1].replace(".pdb", "")

        cumulative_results = list()
        for anchor, lchd_score in zip(anchor_pairs, lchd_scores):
            pair_id1 = f"{pdb_id1}/{primitive_atoms1[anchor[0]].id}"
            pair_id2 = f"{pdb_id2}/{primitive_atoms2[anchor[1]].id}"
            cumulative_results.append((pair_id1, pair_id2, lchd_score))

        if os.path.exists(save_path / out_file_name):
            with open(save_path / out_file_name, "rb") as f:
                cumulative_results += pickle.load(f)

        with open(save_path / out_file_name, "wb") as f:
            pickle.dump(cumulative_results, f)

        time_end = time()

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
