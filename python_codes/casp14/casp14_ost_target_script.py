# Run this inside the OpenStructure Docker container.
# Use the following bash command:
#
# docker run --rm --name ost_run \
# -v $SCRIPT_DIR:/home \
# -v $DATA_DIR:/data \
# registry.scicore.unibas.ch/schwede/openstructure casp14_ost_target_script.py
#
# where $SCRIPT_DIR is the directory containing this script (e.g.: $(pwd)) and
# $DATA_DIR is the directory containing the pickled input data for this script.

import pickle

from ost.io import PDBStrToEntity
from ost.mol.alg import scoring


def main():

    with open("/data/TS427_string_structures.pickle", "rb") as f:
        str_collection = pickle.load(f)

    # A highly nested dict:
    # key1: CASP structure ID (e.g.: T1024)
    # key2: prediction number (1...5)
    # key3: either "single" or "per_residue"
    # key4: score name (lddt, gdtts, rmsd...)
    # key5 (only applicable if we are in "per_residue"): residue ID ([chain name]/[resi number]-[resi tlc])
    # value: score value
    measurement_collection = dict()

    for structure_id, structure_options in str_collection.items():

        reference_structure = PDBStrToEntity(structure_options["true"])
        measurement_collection[structure_id] = dict()

        for predictor_key, structure_str in structure_options.items():

            if predictor_key == "true":
                continue

            predicted_structure = PDBStrToEntity(structure_str)

            scores = scoring.Scorer(
                reference_structure, predicted_structure,
                resnum_alignments=True,  # must be True for CAD score calculation
            )

            single_measurements, per_residue_measurements = dict(), dict()

            single_measurements["lddt"] = scores.lddt
            single_measurements["rmsd"] = scores.rmsd
            single_measurements["cad_score"] = scores.cad_score
            single_measurements["gdtts"] = scores.gdtts
            single_measurements["tm_score"] = scores.tm_score

            per_residue_measurements["lddt"] = {
                f"{resi.chain.name}/{resi.number}-{resi.name}": scores.local_lddt[resi.chain.name][resi.number]
                for resi in predicted_structure.residues
            }

            per_residue_measurements["cad_score"] = {
                f"{resi.chain.name}/{resi.number}-{resi.name}": scores.local_cad_score[resi.chain.name][resi.number]
                for resi in predicted_structure.residues
            }

            measurement_collection[structure_id][int(predictor_key)] = {
                "single": single_measurements,
                "per_residue": per_residue_measurements
            }

            with open("/data/TS427_ost_results.pickle", "wb") as f:
                pickle.dump(measurement_collection, f)


if __name__ == "__main__":
    main()
