from argparse import ArgumentParser, Namespace
from pathlib import Path
from Bio.PDB.PDBParser import PDBParser
from loco_hd import *


def parse_cli_args() -> Namespace:

    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-s1", "--structure1",
        type=str, required=True
    )

    arg_parser.add_argument(
        "-s2", "--structure2",
        type=str, required=True
    )

    arg_parser.add_argument(
        "-pts", "--primitive_typing_scheme",
        type=str, required=True
    )

    args: Namespace = arg_parser.parse_args()

    return args


def prat_to_pra(prat: PrimitiveAtomTemplate) -> PrimitiveAtom:

    resi_id = prat.atom_source.source_residue
    resname = prat.atom_source.source_residue_name
    source = f"{resi_id[2]}/{resi_id[3][1]}-{resname}"

    return PrimitiveAtom(
        prat.primitive_type,
        source,  # this is the tag field!
        prat.coordinates
    )


def main():

    args = parse_cli_args()

    structure1 = PDBParser(QUIET=True).get_structure("s1", args.structure1)
    structure2 = PDBParser(QUIET=True).get_structure("s2", args.structure2)

    primitive_assigner = PrimitiveAssigner(Path(args.primitive_typing_scheme))
    pra_templates1 = primitive_assigner.assign_primitive_structure(structure1)
    pra_templates2 = primitive_assigner.assign_primitive_structure(structure2)

    anchor_pairs = [
        (idx, idx)
        for idx, prat in enumerate(pra_templates1)
        if prat.primitive_type == "Cent"
    ]

    pra1 = list(map(prat_to_pra, pra_templates1))
    pra2 = list(map(prat_to_pra, pra_templates2))

    w_func = WeightFunction("uniform", [3., 10.])
    tag_pairing_rule = TagPairingRule({"accept_same": False})

    lchd = LoCoHD(
        primitive_assigner.all_primitive_types,
        w_func,
        tag_pairing_rule,
        4
    )

    lchd_scores = lchd.from_primitives(
        pra1,
        pra2,
        anchor_pairs,
        10.  # upper distance cutoff at 10 angströms
    )

    print(lchd_scores)


if __name__ == "__main__":
    main()
