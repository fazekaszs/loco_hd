from typing import List

import numpy as np
from scipy.stats import beta
from pathlib import Path

from Bio.PDB import Chain
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Atom import Atom

from loco_hd import LoCoHD, PrimitiveAtom, PrimitiveAssigner, PrimitiveAtomTemplate, WeightFunction, TagPairingRule


def prat_to_pra(prat: PrimitiveAtomTemplate) -> PrimitiveAtom:

    resi_id = prat.atom_source.source_residue
    resname = prat.atom_source.source_residue_name
    source = f"{resi_id[2]}/{resi_id[3][1]}-{resname}"
    return PrimitiveAtom(prat.primitive_type, source, prat.coordinates)


def main():

    protein: Chain = PDBParser(QUIET=True).get_structure("kras", "../data_sources/kras_scan/4obe.pdb")[0]["A"]
    pra_assigner = PrimitiveAssigner(Path("../primitive_typings/all_atom.config.json"))

    prat_list = pra_assigner.assign_primitive_structure(protein)
    pra_list = list(map(prat_to_pra, prat_list))

    scan_anchors: List[PrimitiveAtom] = list()
    atom: Atom
    for atom in protein.get_atoms():

        if atom.name != "CA":
            continue

        new_coords = np.random.uniform(-4, 4, size=(60, 3)) + atom.coord[np.newaxis, :]

        for coord in new_coords:
            current_pra = PrimitiveAtom("dummy", "dummy", coord)
            scan_anchors.append(current_pra)

    gdp_molecule = protein[("H_GDP", 201, " ")]
    ref_coord = gdp_molecule["O2\'"].coord
    ref_anchor = [PrimitiveAtom("dummy", "dummy", ref_coord), ]

    anchor_idxs = [(0, idx) for idx in range(len(scan_anchors))]

    weight_function = WeightFunction("uniform", [3, 10])
    tag_pairing_rule = TagPairingRule({"accept_same": False})
    lchd = LoCoHD(pra_assigner.all_primitive_types + ["dummy"], weight_function, tag_pairing_rule)
    lchd_scores = lchd.from_primitives(ref_anchor + pra_list, scan_anchors + pra_list, anchor_idxs, 10.)
    lchd_scores = beta.cdf(lchd_scores, 10.52, 33.48)

    out = ""
    scan_pra: PrimitiveAtom
    for idx, (scan_pra, score) in enumerate(zip(scan_anchors, lchd_scores)):

        line = "ATOM  "
        line += f"{idx:5d} "  # serial + space
        line += " D  "  # name
        line += " "  # altLoc
        line += "DUM "  # resName + space
        line += "A"  # chainID
        line += "   1"  # resSeq
        line += "    "  # iCode + 3 spaces
        line += f"{scan_pra.coordinates[0]:8.3f}"  # x
        line += f"{scan_pra.coordinates[1]:8.3f}"  # y
        line += f"{scan_pra.coordinates[2]:8.3f}"  # z
        line += "      "  # occupancy
        line += f"{score:6.2f}"
        line += "\n"

        out += line

    with open("../workdir/kras_scan/4obe_scan.pdb", "w") as f:
        f.write(out)


if __name__ == "__main__":
    main()
