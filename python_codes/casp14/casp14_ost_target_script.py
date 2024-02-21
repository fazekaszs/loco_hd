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
import os

from multiprocessing import Pool
from datetime import datetime

from ost.io import PDBStrToEntity
from ost.mol.alg import scoring


def add_log_line(msg: str) -> None:

    dt_now = datetime.now().strftime("[ %Y.%m.%d %H:%M:%S ]")
    log_line = f"{dt_now} {msg}\n"

    with open(f"/data/ost_results.log", "a") as f:
        f.write(log_line)


def _run_on_structure_pair(structure_id, ref_structure_str, pred_structure_str):

    reference_structure = PDBStrToEntity(ref_structure_str)
    predicted_structure = PDBStrToEntity(pred_structure_str)

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

    return structure_id, single_measurements, per_residue_measurements


def run_on_structure_pair(args):
    return _run_on_structure_pair(*args)


def main():

    if os.path.exists("/data/ost_results.log"):
        os.remove("/data/ost_results.log")

    with open(f"/data/filtered_structures.pickle", "rb") as f:
        str_collection = pickle.load(f)

    # A highly nested dict:
    # key1: CASP structure name (e.g.: T1024)
    # key2: prediction number (structure_id) (1...5)
    # key3: either "single" or "per_residue"
    # key4: score name (lddt, gdtts, rmsd...)
    # key5 (only applicable if we are in "per_residue"): residue ID ([chain name]/[resi number]-[resi tlc])
    # value: score value
    measurement_collection = dict()

    for idx, (structure_name, structure_bundle) in enumerate(str_collection.items()):

        add_log_line(f"Starting structure bundle {structure_name} ({idx + 1}/{len(str_collection)})")

        measurement_collection[structure_name] = dict()

        # Create arguments for the multiprocessing mapping.
        structure_pairs = [
            (structure_id, structure_bundle["true"], pred_structure_str)
            for structure_id, pred_structure_str in structure_bundle.items()
            if structure_id != "true"
        ]

        # Perform task with multiprocessing.
        n_of_processes = min([len(structure_pairs), 8])

        add_log_line(f"Starting multiprocessing with {n_of_processes} number of processes")
        with Pool(n_of_processes) as pool:
            results = pool.map(run_on_structure_pair, structure_pairs)

        # Update the measurement_collection dictionary.

        add_log_line(f"Updating measurement collection")
        for structure_id, single_measurements, per_residue_measurements in results:

            measurement_collection[structure_name][int(structure_id)] = {
                "single": single_measurements,
                "per_residue": per_residue_measurements
            }

    with open(f"/data/ost_results.pickle", "wb") as f:
        pickle.dump(measurement_collection, f)


if __name__ == "__main__":
    main()
