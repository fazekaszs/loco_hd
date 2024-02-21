import sys
import os
import pickle

from config import (
    PREDICTOR_KEY,
    PRED_TAR_DIR,
    REFS_TAR_DIR,
    EXTRACTOR_OUTPUT_DIR,
    IGNORED_STRUCTURES,
)

from tarfile_structure_extractor_utils import (
    read_in_ref_structures,
    read_in_pred_structures,
    normalize_ref_structures,
    rename_nameless_chains,
    repair_chain_correspondence,
    apply_common_resi_filtering,
    apply_common_id_filtering,
    count_identical_chains,
    stringify_structures
)

# Do only a test run and don't save anything.
TEST_RUN = False


def main():

    # getting the structures from the databases (tar-files)
    all_structures = read_in_ref_structures(REFS_TAR_DIR, IGNORED_STRUCTURES)
    all_structures = read_in_pred_structures(all_structures, PRED_TAR_DIR, PREDICTOR_KEY)

    # in-place, remove disordered elements and non-canonical amino acids
    normalize_ref_structures(all_structures)

    # in-place, replaces " " chain IDs with "A"
    rename_nameless_chains(all_structures)

    # in-place, pairs up reference and predicted chain IDs based on sequence alignments
    repair_chain_correspondence(all_structures)

    # in-place, removes residues that are not present in all bundles
    apply_common_resi_filtering(all_structures)

    # in-place, removes bundle-level non-common entities recursively
    apply_common_id_filtering(all_structures)

    # only reports: it counts the (almost) identical chains
    count_identical_chains(all_structures, 0.8)

    # converts the BioPython Structure objects to pdb strings
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
