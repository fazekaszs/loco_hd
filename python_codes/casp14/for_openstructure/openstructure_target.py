import pickle

from ost.io import PDBStrToEntity
from ost.mol.alg import scoring


def main():

    with open("/data/TS427_strs.pickle", "rb") as f:
        str_collection = pickle.load(f)

    for structure_id, structure_options in str_collection.items():

        reference_structure = PDBStrToEntity(structure_options["true"])
        for predictor_key, structure_str in structure_options.items():

            if predictor_key == "true":
                continue

            predicted_structure = PDBStrToEntity(structure_str)
            scores = scoring.Scorer(
                reference_structure, predicted_structure,
                # resnum_alignments=True,  # must be True for CAD score calculation
            )

            full_key = f"{structure_id}_{predictor_key}"

            with open("/data/TS427_ost_results.tsv", "a") as f:
                f.write(f"{full_key}\t")
                f.write(f"lDDT: {scores.lddt}\t")
                f.write(f"TM-score: {scores.tm_score}\t")
                # f.write(f"CAD-score: {scores.cad_score}\t")
                f.write("\n")


if __name__ == "__main__":
    main()
