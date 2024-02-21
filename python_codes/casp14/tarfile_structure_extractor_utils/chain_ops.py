import numpy as np

from typing import Dict, List, Tuple, Set

from Bio.Align import PairwiseAligner
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain

from scipy.optimize import linear_sum_assignment

_TLC_TO_OLC = {
    "GLY": "G", "ALA": "A", "VAL": "V", "ILE": "I", "LEU": "L",
    "PHE": "F", "SER": "S", "THR": "T", "TYR": "Y", "ASP": "D",
    "GLU": "E", "ASN": "N", "GLN": "Q", "CYS": "C", "MET": "M",
    "PRO": "P", "LYS": "K", "ARG": "R", "TRP": "W", "HIS": "H",
}


def rename_nameless_chains(all_structures: Dict[str, Dict[str, Structure]]) -> None:

    for structure_name, structure_bundle in all_structures.items():

        current_chains: Dict[str, Chain] = structure_bundle["true"][0].child_dict

        print(f"{structure_name} contains {len(current_chains)} chain(s)!")

        if len(current_chains) != 1:
            continue

        chain_id, chain_obj = list(current_chains.items())[0]

        print(f"Chain ID is {chain_id}!")

        if chain_id != " ":
            continue

        for structure in list(structure_bundle.values()):
            structure[0][" "].id = "A"  # Set new chain ID


def _chain_to_str_sequence(chain: Chain) -> str:
    return "".join((_TLC_TO_OLC[r.resname] for r in chain.child_list))


def _pair_up_chains(
    ref_chains: List[Tuple[str, str]],
    pred_chains: List[Tuple[str, str]]
) -> Tuple[List[Tuple[str, str]], np.ndarray, np.ndarray]:

    seq_seq_alignment_mx = list()
    aligner = PairwiseAligner()

    for ref_chain_id, ref_chain_seq in ref_chains:

        seq_seq_alignment_mx.append(list())
        for pred_chain_id, pred_chain_seq in pred_chains:

            # Score is the number of correctly aligned residues.
            alignment_score = aligner.align(ref_chain_seq, pred_chain_seq).score

            # Normalize it to a max of 1.
            alignment_score /= max([len(ref_chain_seq), len(pred_chain_seq), ])

            seq_seq_alignment_mx[-1].append(alignment_score)

    seq_seq_alignment_mx = np.array(seq_seq_alignment_mx)
    row_idxs, col_idxs = linear_sum_assignment(seq_seq_alignment_mx, maximize=True)

    chain_id_pairs = [
        (ref_chains[ref_idx][0], pred_chains[pred_idx][0])
        for ref_idx, pred_idx in zip(row_idxs, col_idxs)
    ]

    mean_score = np.mean(seq_seq_alignment_mx[row_idxs, col_idxs])
    min_score = np.min(seq_seq_alignment_mx[row_idxs, col_idxs])

    return chain_id_pairs, mean_score, min_score


def repair_chain_correspondence(all_structures: Dict[str, Dict[str, Structure]]):

    for structure_name, structure_bundle in all_structures.items():

        # Get the chains and their sequences in the reference structure.
        ref_sequences: List[Tuple[str, str]] = [
            (chain_id, _chain_to_str_sequence(chain))
            for chain_id, chain
            in structure_bundle["true"][0].child_dict.items()
        ]

        # No need to deal with chain correspondence: there is only one chain.
        if len(ref_sequences) == 1:
            continue

        for structure_id, structure in structure_bundle.items():

            if structure_id == "true":
                continue

            # Get the chains and their sequences in the current predicted structure.
            pred_sequences: List[Tuple[str, str]] = [
                (chain_id, _chain_to_str_sequence(chain))
                for chain_id, chain
                in structure[0].child_dict.items()
            ]

            # Search for best chain pairing based on the sequences.
            pairing, mean_score, min_score = _pair_up_chains(ref_sequences, pred_sequences)

            print(f"Chain pairing for {structure_name}_{structure_id}: {pairing}")
            print(f"Mean optimal pairing score: {mean_score:.2%}, min.: {min_score:.2%}")

            # Rename chains to their correct chain ID + a star annotation.
            for ref_chain_id, pred_chain_id in pairing:
                structure[0][pred_chain_id].id = ref_chain_id + "*"

            # Remove chains that were not annotated with stars.
            # These are "unassigned jobs and workers" in the linear sum assignment problem.
            for pred_chain_id in list(structure[0].child_dict.keys()):
                if pred_chain_id.endswith("*"):
                    continue
                structure[0].detach_child(pred_chain_id)
                print(f"Chain ID {pred_chain_id} was unassigned and removed!")

            # Remove the chain "*" annotations.
            for pred_chain_id in list(structure[0].child_dict.keys()):
                structure[0][pred_chain_id].id = pred_chain_id[:-1]


def _n_of_identical_chains(ref_chains: List[Tuple[str, str]], threshold: float) -> None:
    aligner = PairwiseAligner()
    identical_chains: List[Tuple[str, Set[str]]] = list()

    for ref_chain_id, ref_chain_seq in ref_chains:

        for test_seq, chain_id_set in identical_chains:

            # Score is the number of correctly aligned residues.
            alignment_score = aligner.align(ref_chain_seq, test_seq).score

            # Normalize it to a max of 1.
            alignment_score /= max([len(ref_chain_seq), len(test_seq), ])

            # Continue if the sequence does not match with the current test sequence.
            if alignment_score < threshold:
                continue

            # There is a sequence match! Add the chain ID to the current set and break out the loop.
            chain_id_set.add(ref_chain_id)
            break

        else:
            identical_chains.append((ref_chain_seq, {ref_chain_id}))

    # Report identical chains.
    out_str = "Identical chain groups: "
    for idx, (_, chain_id_set) in enumerate(identical_chains):
        out_str += f"Group {idx}: " + "".join(chain_id_set) + ", "
    out_str = out_str[:-2]
    print(out_str)


def count_identical_chains(all_structures: Dict[str, Dict[str, Structure]], threshold: float) -> None:

    print(f"Counting identical chains! I am using a threshold of {threshold:.2%}!")

    for structure_name, structure_bundle in all_structures.items():

        print(f"Counting identical chains for {structure_name}:")

        # Get the chains and their sequences in the reference structure.
        ref_sequences: List[Tuple[str, str]] = [
            (chain_id, _chain_to_str_sequence(chain))
            for chain_id, chain
            in structure_bundle["true"][0].child_dict.items()
        ]

        _n_of_identical_chains(ref_sequences, threshold)
