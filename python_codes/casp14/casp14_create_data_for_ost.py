# Convert the pickled file saved from casp14_tarfile_structure_extractor.py into a
# nested dictionary of PDB strings. The two-level nested dictionary has keys
# of 1. structure id (in CASP format) and 2. "true"/"1"/"2"... (a predictor_key).

import pickle
import os

from pathlib import Path
from typing import Dict

from Bio.PDB.Structure import Structure
from Bio.PDB.PDBIO import PDBIO

# Set the necessary constants. The available predictor keys are the following:
# AF2: TS427, BAKER: TS473, BAKER-experimental: TS403, FEIG-R2: TS480, Zhang: TS129
PREDICTOR_KEY = "TS427"
SOURCE_DIR = Path(f"../workdir/casp14/{PREDICTOR_KEY}_results")
TARGET_DIR = Path(f"../workdir/casp14/for_openstructure/{PREDICTOR_KEY}_results")


class FakeFileHandler:

    def __init__(self):
        self.content = ""

    def write(self, other: str):
        self.content += other

    def close(self):
        pass


def main():

    pdb_io = PDBIO()

    structure_collection_path = SOURCE_DIR / f"{PREDICTOR_KEY}_structures.pickle"
    with open(structure_collection_path, "rb") as f:
        structure_collection: Dict[str, Dict[str, Structure]] = pickle.load(f)

    out: Dict[str, Dict[str, str]] = dict()

    for structure_id, structure_options in structure_collection.items():

        out[structure_id] = dict()

        for predictor_key, structure in structure_options.items():

            # Set the chain ID or else OpenStructure will fail to run the scoring
            chain = structure[0].child_list[0]
            chain.id = "A"

            ffh = FakeFileHandler()

            pdb_io.set_structure(structure)
            pdb_io.save(ffh)

            out[structure_id][predictor_key] = ffh.content
            print(f"Structure {structure_id}_{predictor_key} done!")

    if not os.path.exists(TARGET_DIR):
        os.mkdir(TARGET_DIR)

    str_collection_path = TARGET_DIR / f"{PREDICTOR_KEY}_strs.pickle"
    with open(str_collection_path, "wb") as f:
        pickle.dump(out, f)


if __name__ == "__main__":
    main()
