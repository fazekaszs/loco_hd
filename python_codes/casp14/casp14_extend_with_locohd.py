import pickle
import sys
from typing import Dict
from pathlib import Path

import numpy as np

from loco_hd import LoCoHD, PrimitiveAtom, WeightFunction, PrimitiveAssigner, PrimitiveAtomTemplate, TagPairingRule
from tarfile_structure_extractor_utils import InLinePDBParser
from config import EXTRACTOR_OUTPUT_DIR


def prat_to_pra(prat: PrimitiveAtomTemplate) -> PrimitiveAtom:

    resi_id = prat.atom_source.source_residue
    resname = prat.atom_source.source_residue_name
    source = f"{resi_id[2]}/{resi_id[3][1]}-{resname}"
    return PrimitiveAtom(prat.primitive_type, source, prat.coordinates)


def main():

    # Load Structures
    with open(EXTRACTOR_OUTPUT_DIR / f"filtered_structures.pickle", "rb") as f:
        structure_collection: Dict[str, Dict[str, str]] = pickle.load(f)

    # Load scores
    with open(EXTRACTOR_OUTPUT_DIR / f"ost_results.pickle", "rb") as f:
        score_collection = pickle.load(f)

    # Create the primitive assigner
    primitive_assigner = PrimitiveAssigner(Path("../../primitive_typings/all_atom_with_centroid.config.json"))

    # Create the LoCoHD instance.
    w_func = WeightFunction("uniform", [3, 10])
    tag_pairing_rule = TagPairingRule({"accept_same": False})
    lchd = LoCoHD(primitive_assigner.all_primitive_types, w_func, tag_pairing_rule)

    # Create in-line PDB parser.
    pdb_parser = InLinePDBParser(QUIET=True)

    for structure_name, structure_bundle in structure_collection.items():

        # Transform the real structure into a list of primitive atoms
        # and get the anchors simultaneously.
        reference_structure = pdb_parser.from_str("", structure_bundle["true"])
        true_pra_templates = primitive_assigner.assign_primitive_structure(reference_structure)
        anchors = [(idx, idx) for idx, prat in enumerate(true_pra_templates) if prat.primitive_type == "Cent"]
        true_prim_atoms = list(map(prat_to_pra, true_pra_templates))

        for structure_id, structure_str in structure_bundle.items():

            if structure_id == "true":
                continue

            current_structure = pdb_parser.from_str("", structure_str)

            # Transform the predicted structure into a list of primitive atoms.
            pred_pra_templates = primitive_assigner.assign_primitive_structure(current_structure)
            pred_prim_atoms = list(map(prat_to_pra, pred_pra_templates))

            # Anchor correspondence sanity check.
            for idx1, idx2 in anchors:

                tag1 = true_prim_atoms[idx1].tag
                tag2 = pred_prim_atoms[idx2].tag

                if tag1 == tag2:
                    continue

                sys.exit(
                    f"ERROR: tag mismatch! "
                    f"{tag1} and {tag2} are not equal in structure {structure_name}_{structure_id}!"
                )

            # Calculate LoCoHD score with a threshold_distance = 10.
            lchd_scores = lchd.from_primitives(true_prim_atoms, pred_prim_atoms, anchors, 10)

            # Extract anchor atom parent residue IDs from the primitive atom tag fields.
            anchor_resi_ids = [true_prim_atoms[anchor[0]].tag for anchor in anchors]

            # Refresh the score collection
            score_collection[structure_name][int(structure_id)]["per_residue"]["LoCoHD"] = {
                resi_id: score for resi_id, score in zip(anchor_resi_ids, lchd_scores)
            }
            score_collection[structure_name][int(structure_id)]["single"]["LoCoHD"] = np.mean(lchd_scores)

            print(f"\r{structure_name}, prediction {structure_id} done...", end="")

    with open(EXTRACTOR_OUTPUT_DIR / f"ost_results_extended.pickle", "wb") as f:
        pickle.dump(score_collection, f)
    print("\rAll done!")


if __name__ == "__main__":
    main()
