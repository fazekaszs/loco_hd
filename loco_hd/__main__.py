import json
from typing import List, Tuple, FrozenSet, Dict, Optional
from argparse import ArgumentParser, Namespace
from pathlib import Path
from Bio.PDB.PDBParser import PDBParser
from loco_hd import *

# chain ID (eg.: A), resi ID (eg.: 123-GLY), atom set (eg.: {CG, CZ})
TagIDType = Tuple[str, str, FrozenSet[str]]


def parse_anchor_pairing(anchor_pairing_str_list: List[str]) -> List[Tuple[TagIDType, TagIDType]]:

    anchor_pairings = list()
    for anchor_tag_pair in anchor_pairing_str_list:

        anchor_tag1, anchor_tag2 = anchor_tag_pair.split(":")

        chain_id1, resi_id1, atom_set1 = anchor_tag1.split("/")
        atom_set1 = frozenset(atom_set1.split(","))

        chain_id2, resi_id2, atom_set2 = anchor_tag2.split("/")
        atom_set2 = frozenset(atom_set2.split(","))

        anchor_tag1 = (chain_id1, resi_id1, atom_set1)
        anchor_tag2 = (chain_id2, resi_id2, atom_set2)

        anchor_pairings.append((anchor_tag1, anchor_tag2))

    return anchor_pairings


def pra_template_list_to_idx_dict(pra_templates: List[PrimitiveAtomTemplate]) -> Dict[TagIDType, int]:

    pra_id_to_idx = dict()
    for idx, pra_template in enumerate(pra_templates):

        pra_source = pra_template.atom_source

        chain_id = pra_source.source_residue[2]
        resi_id = f"{pra_source.source_residue[3][1]}-{pra_source.source_residue_name}"
        atom_set = frozenset(pra_source.source_atom)

        pra_source_id = (chain_id, resi_id, atom_set)  # unique ID for the primitive atom template
        pra_id_to_idx[pra_source_id] = idx

    return pra_id_to_idx


def parse_cli_args() -> Namespace:

    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "-s1", "--structure1",
        type=str, required=True,
        help="Path to the first pdb file to be compared."
    )

    arg_parser.add_argument(
        "-s2", "--structure2",
        type=str, required=True,
        help="Path to the second pdb file to be compared."
    )

    arg_parser.add_argument(
        "-pts", "--primitive_typing_scheme",
        type=str, required=True,
        help="Path to the primitive typing scheme json file."
    )

    # Argument for the anchor pairing.
    arg_parser.add_argument(
        "-apf", "--anchor_pairing_file",
        type=Path,
        required=True,
        help="A path pointing to an anchor pairing file. This should be a textfile "
             "enumerating the identifier of anchor pairs. An anchor pair is defined by "
             "the following example string: A/123-TYR/CG,CZ:B/45-ALA/CB which would match "
             "up a primitive atom from chain A, Tyr 123, atom set {CG, CZ}, and a primitive atom "
             "from chain B, Ala 45, atom set {CB}. Atom sets are necessary, since primitive atoms "
             "can come from multiple normal atoms (like residue centroids or coarse grained atoms). "
             "These individual pairings should be separated by semicolons."
    )

    arg_parser.add_argument(
        "-mn", "--model_number",
        type=int, default=0,
        help="The model number in the pdb files to be compared (0 by default)."
    )

    arg_parser.add_argument(
        "-nt", "--number_of_threads",
        type=Optional[int], default=None,
        help="Specifies the number of threads used by the LoCoHD instance. "
             "By default, it uses all available threads."
    )

    arg_parser.add_argument(
        "-udc", "--upper_distance_cutoff",
        type=float, default=10.,
        help="Specifies the upper distance cutoff during LoCoHD calculations."
    )

    # Argument for the TagPairingRule object.
    tpra_default = "{\"accept_same\": false}"
    arg_parser.add_argument(
        "-tpra", "--tag_pairing_rule_args",
        type=str,
        default=tpra_default,
        help=f"A json formatted string that is parsed into a dictionary "
             f"and then used to initialize the TagPairingRule. "
             f"It is '{tpra_default}' by default."
    )

    # Argument for the WeightFunction object.
    wfa_default = "{\"function_name\": \"uniform\", \"parameters\": [3.0, 10.0]}"
    arg_parser.add_argument(
        "-wfa", "--weight_function_args",
        type=str,
        default=wfa_default,
        help=f"A json formatted string that is parsed into a dictionary "
             f"and then used to initialize the WeightFunction. "
             f"It is '{wfa_default}' by default."
    )

    args: Namespace = arg_parser.parse_args()

    # Post-processing
    args.tag_pairing_rule_args = json.loads(args.tag_pairing_rule_args)
    args.weight_function_args = json.loads(args.weight_function_args)

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

    # Read in the anchor pairing.
    with open(args.anchor_pairing_file, "r") as f:
        anchor_pairing_str_list = f.read().replace("\n", "").split(";")

    anchor_pairing = parse_anchor_pairing(anchor_pairing_str_list)

    # Read in structures.
    structure1 = PDBParser(QUIET=True).get_structure("s1", args.structure1)[args.model_number]
    structure2 = PDBParser(QUIET=True).get_structure("s2", args.structure2)[args.model_number]

    # Read in the primitive type assigner.
    primitive_assigner = PrimitiveAssigner(Path(args.primitive_typing_scheme))

    # Create primitive atom templates.
    pra_templates1 = primitive_assigner.assign_primitive_structure(structure1)
    pra_templates2 = primitive_assigner.assign_primitive_structure(structure2)

    # Identify anchor atoms.
    prat_id_to_idx1 = pra_template_list_to_idx_dict(pra_templates1)
    prat_id_to_idx2 = pra_template_list_to_idx_dict(pra_templates2)

    anchor_pairs = [
        (prat_id_to_idx1[anchor_id1], prat_id_to_idx2[anchor_id2])
        for anchor_id1, anchor_id2
        in anchor_pairing
    ]

    # Create primitive atoms from primitive atom templates.
    pra1 = list(map(prat_to_pra, pra_templates1))
    pra2 = list(map(prat_to_pra, pra_templates2))

    # Initialize the LoCoHD object.
    w_func = WeightFunction(**args.weight_function_args)
    tag_pairing_rule = TagPairingRule(args.tag_pairing_rule_args)

    lchd = LoCoHD(
        primitive_assigner.all_primitive_types,
        w_func,
        tag_pairing_rule,
        args.number_of_threads
    )

    # Run calculation.
    lchd_scores = lchd.from_primitives(
        pra1,
        pra2,
        anchor_pairs,
        args.upper_distance_cutoff
    )

    for anchor_pair_str, score in zip(anchor_pairing_str_list, lchd_scores):
        print(f"LoCoHD({anchor_pair_str}) = {score}")


if __name__ == "__main__":
    main()
