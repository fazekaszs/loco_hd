import numpy as np

from typing import Dict, List, Tuple

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


def _chain_to_str_sequence(chain: Chain) -> str:
    return "".join((_TLC_TO_OLC[r.resname] for r in chain.child_list))


def _pair_up_chains(
    ref_chains: List[Tuple[str, str]],
    pred_chains: List[Tuple[str, str]]
) -> List[Tuple[str, str]]:

    seq_seq_alignment_mx = list()
    aligner = PairwiseAligner()

    for ref_chain_id, ref_chain_seq in ref_chains:

        seq_seq_alignment_mx.append(list())
        for pred_chain_id, pred_chain_seq in pred_chains:

            alignment_score = aligner.align(ref_chain_seq, pred_chain_seq).score
            seq_seq_alignment_mx[-1].append(alignment_score)

    seq_seq_alignment_mx = np.array(seq_seq_alignment_mx)
    row_idxs, col_idxs = linear_sum_assignment(seq_seq_alignment_mx, maximize=True)

    chain_id_pairs = [
        (ref_chains[ref_idx][0], pred_chains[pred_idx][0])
        for ref_idx, pred_idx in zip(row_idxs, col_idxs)
    ]

    return chain_id_pairs


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
            pairing = _pair_up_chains(ref_sequences, pred_sequences)

            print(f"Chain pairing for {structure_name}_{structure_id}: {pairing}")

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
