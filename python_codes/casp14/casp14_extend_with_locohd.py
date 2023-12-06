import pickle
from typing import Dict
from pathlib import Path

import numpy as np
from Bio.PDB.Structure import Structure

from loco_hd import LoCoHD, PrimitiveAtom, WeightFunction, PrimitiveAssigner, PrimitiveAtomTemplate, TagPairingRule

PREDICTOR_KEY = "TS427"
WORKDIR = Path(f"../../workdir/casp14/{PREDICTOR_KEY}_results/")


def prat_to_pra(prat: PrimitiveAtomTemplate) -> PrimitiveAtom:

    resi_id = prat.atom_source.source_residue
    resname = prat.atom_source.source_residue_name
    source = f"{resi_id[2]}/{resi_id[3][1]}-{resname}"
    return PrimitiveAtom(prat.primitive_type, source, prat.coordinates)


def main():

    # Load Structures
    with open(WORKDIR / f"{PREDICTOR_KEY}_biopython_structures.pickle", "rb") as f:
        structure_collection: Dict[str, Dict[str, Structure]] = pickle.load(f)

    # Load scores
    with open(WORKDIR / f"{PREDICTOR_KEY}_ost_results.pickle", "rb") as f:
        score_collection = pickle.load(f)

    # Create the primitive assigner
    primitive_assigner = PrimitiveAssigner(Path("../../primitive_typings/all_atom_with_centroid.config.json"))

    # Create the LoCoHD instance.
    w_func = WeightFunction("uniform", [3, 10])
    tag_pairing_rule = TagPairingRule({"accept_same": False})
    lchd = LoCoHD(primitive_assigner.all_primitive_types, w_func, tag_pairing_rule)

    for structure_id, structure_options in structure_collection.items():

        # Transform the real structure into a list of primitive atoms
        # and get the anchors simultaneously.
        reference_structure = structure_options["true"]
        true_pra_templates = primitive_assigner.assign_primitive_structure(reference_structure)
        anchors = [(idx, idx) for idx, prat in enumerate(true_pra_templates) if prat.primitive_type == "Cent"]
        true_prim_atoms = list(map(prat_to_pra, true_pra_templates))

        for predictor_key, current_structure in structure_options.items():

            if predictor_key == "true":
                continue

            # Transform the predicted structure into a list of primitive atoms.
            pred_pra_templates = primitive_assigner.assign_primitive_structure(current_structure)
            pred_prim_atoms = list(map(prat_to_pra, pred_pra_templates))

            # Calculate LoCoHD score with a threshold_distance = 10.
            lchd_scores = lchd.from_primitives(true_prim_atoms, pred_prim_atoms, anchors, 10)

            # Replace the chain id " " in the primitive atom tags to "A"
            anchor_resi_ids = ["A" + true_prim_atoms[anchor[0]].tag[1:] for anchor in anchors]

            # Refresh the score collection
            score_collection[structure_id][int(predictor_key)]["per_residue"]["LoCoHD"] = {
                resi_id: score for resi_id, score in zip(anchor_resi_ids, lchd_scores)
            }
            score_collection[structure_id][int(predictor_key)]["single"]["LoCoHD"] = np.mean(lchd_scores)

            print(f"\r{structure_id}, prediction {predictor_key} done...", end="")

    with open(WORKDIR / f"{PREDICTOR_KEY}_ost_results_extended.pickle", "wb") as f:
        pickle.dump(score_collection, f)
    print("\rAll done!")


if __name__ == "__main__":
    main()