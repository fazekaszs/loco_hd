import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Structure import Structure

from loco_hd import (
    PrimitiveAssigner,
    LoCoHD,
    TagPairingRule,
    WeightFunction,
    PrimitiveAtomTemplate,
    PrimitiveAtom
)

MUTANT_PDBS = Path("../../data_sources/for_mutation_tests/mutants")
REF_STRUCTURE = Path("../../data_sources/for_mutation_tests/Optimized_1cho.pdb")
PRIMITIVE_TYPING_SCHEME = "all_atom_with_centroid.config.json"


def prat_to_pra(prat: PrimitiveAtomTemplate) -> PrimitiveAtom:

    resi_id = prat.atom_source.source_residue
    resname = prat.atom_source.source_residue_name
    source = f"{resi_id[2]}/{resi_id[3][1]}-{resname}"
    return PrimitiveAtom(prat.primitive_type, source, prat.coordinates)


def main():

    assigner = PrimitiveAssigner(Path("../../primitive_typings") / PRIMITIVE_TYPING_SCHEME)
    tag_pairing_rule = TagPairingRule({"accept_same": False})
    weight_function = WeightFunction("uniform", [3., 10.])
    locohd = LoCoHD(assigner.all_primitive_types, weight_function, tag_pairing_rule)

    pdb_parser = PDBParser(QUIET=True)

    exp_structure: Structure = pdb_parser.get_structure("ref", REF_STRUCTURE)
    exp_structure_prats = assigner.assign_primitive_structure(exp_structure)
    exp_structure_pras = list(map(prat_to_pra, exp_structure_prats))
    cent_lookup = {
        pra.tag.split("-")[0]: idx
        for idx, pra in enumerate(exp_structure_pras)
        if pra.primitive_type == "Cent"
    }

    all_scores = list()

    mut_filenames = list(filter(lambda x: x.endswith(".pdb"), os.listdir(MUTANT_PDBS)))
    for mut_filename in mut_filenames:

        mut_structure_id = mut_filename.replace(".pdb", "")
        mut_structure: Structure = pdb_parser.get_structure(mut_structure_id, MUTANT_PDBS / mut_filename)
        mut_structure_prats = assigner.assign_primitive_structure(mut_structure)
        mut_structure_pras = list(map(prat_to_pra, mut_structure_prats))

        anchors = [
            (idx, cent_lookup[pra.tag.split("-")[0]])
            for idx, pra in enumerate(mut_structure_pras)
            if pra.primitive_type == "Cent"
        ]
        locohd_scores = locohd.from_primitives(
            mut_structure_pras, exp_structure_pras, anchors, 10.
        )
        all_scores.append(locohd_scores)
        print(f"{mut_filename} done!")

    all_scores = np.array(all_scores)
    plt.plot(np.max(all_scores, axis=0))
    plt.show()


if __name__ == "__main__":
    main()
