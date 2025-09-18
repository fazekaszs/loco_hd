import numpy as np
import itertools
import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple

from loco_hd import WeightFunction, LoCoHD, PrimitiveAtom, StatisticalDistance


PRIMITIVE_TYPES = [
    "A", "B", "C", "D", "E"
]
THRESHOLD_DISTANCE = 50.
DATA_DIR = Path("tests/test_data")
VARY_STATISTICAL_DISTANCES = True


def generate_wf_params(counts: Tuple[int, ...] = (5, 5, 10, 10, 10)):

    # Generate random weight functions

    weight_functions = list()

    # hyper_exp, two parameter type
    for _ in range(counts[0]):
        param_b = 1. / np.random.uniform(3., 20.)
        weight_functions.append((
            "hyper_exp", [1., param_b]
        ))
    
    # hyper_exp, six parameter type
    for _ in range(counts[1]):
        param_a = np.random.uniform(1E-5, 1., size=3)
        param_b = 1. / np.random.uniform(3., 20., size=3)
        weight_functions.append((
            "hyper_exp", np.concatenate([param_a, param_b]).tolist()
        ))
    
    # dagum
    for _ in range(counts[2]):
        param_a = np.random.uniform(0.5, 3.0)
        param_b = np.random.uniform(1.5, 4.0)
        param_c = np.random.uniform(1., 25.)
        weight_functions.append((
            "dagum", [param_a, param_b, param_c]
        ))
    
    # uniform
    for _ in range(counts[3]):
        param_a = np.random.uniform(0.5, 10.0)
        delta_ab = np.random.uniform(0.1, 10.0)
        weight_functions.append((
            "uniform", [param_a, param_a + delta_ab]
        ))
    
    # kumaraswamy
    for _ in range(counts[4]):
        param_a = np.random.uniform(0.5, 10.0)
        delta_ab = np.random.uniform(0.1, 10.0)
        param_c = np.random.uniform(1. + 1E-5, 10.)
        param_d = np.random.uniform(1. + 1E-5, 10.)
        weight_functions.append((
            "kumaraswamy", [param_a, param_a + delta_ab, param_c, param_d]
        ))

    return weight_functions


def generate_statistical_distances(counts: Tuple[int, ...] = (5, 5, 5)):

    # Generate randomly parametrized statistical distances

    statistical_distances = list()

    # Hellinger
    for _ in range(counts[0]):
        exponent = np.random.uniform(1., 5.)
        statistical_distances.append(
            ("Hellinger", [exponent, ])
        )

    # Kolmogorov-Smirnov
    statistical_distances.append(
        ("Kolmogorov-Smirnov", [])
    )

    # Kullback-Leibler
    for _ in range(counts[1]):
        epsilon = np.random.uniform(0.001, 5.)
        statistical_distances.append(
            ("Kullback-Leibler", [epsilon, ])
        )

    # Renyi
    for _ in range(counts[2]):
        alpha = np.random.uniform(0.001, 5.)
        epsilon = np.random.uniform(0.001, 5.)
        statistical_distances.append(
            ("Renyi", [alpha, epsilon])
        )

    return statistical_distances


def generate_sequence_coord_pairs(n_points: int = 40):

    seq_coord_pairs = list()

    for _ in range(n_points):

        seq_len = np.random.randint(50, 301)
        seq = np.random.choice(PRIMITIVE_TYPES, seq_len, replace=True).tolist()
        coords = np.random.uniform(-50, 50, size=(seq_len, 3)).tolist()
        seq_coord_pairs.append({
            "seq": seq,
            "coords": coords
        })

    return seq_coord_pairs


def main():

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    if not VARY_STATISTICAL_DISTANCES:
        wfs = generate_wf_params()
        sds = [("Hellinger", [2., ]), ]
        scps = generate_sequence_coord_pairs()
    else:
        wfs = generate_wf_params(counts=(2, 2, 4, 4, 4))
        sds = generate_statistical_distances()
        scps = generate_sequence_coord_pairs(n_points=15)

    input_collection = {
        "primitive_types": PRIMITIVE_TYPES,
        "threshold_distance": THRESHOLD_DISTANCE,
        "weight_functions": wfs,
        "statistical_distances": sds,
        "sequence_coordinate_pairs": scps
    }


    with open(DATA_DIR / f"test_input_collection_{timestamp}.pickle", "wb") as f:
        pickle.dump(input_collection, f)

    iterator = itertools.product(
        enumerate(wfs),
        enumerate(sds),
        enumerate(scps), 
        enumerate(scps)
    )

    collection = dict()
    
    for (idx_wf, wf), (idx_sd, sd), (idx1, scp1), (idx2, scp2) in iterator:

        if idx1 == idx2:
            continue
        
        lchd = LoCoHD(
            categories=PRIMITIVE_TYPES,
            w_func=WeightFunction(*wf),
            statistical_distance=StatisticalDistance(*sd)
        )
        pras1 = [PrimitiveAtom(x, "", y) for x, y in zip(scp1["seq"], scp1["coords"])]
        pras2 = [PrimitiveAtom(x, "", y) for x, y in zip(scp2["seq"], scp2["coords"])]
        anchors = [(x, x) for x in range(min(len(pras1), len(pras2)))]
        scores = lchd.from_primitives(pras1, pras2, anchors, THRESHOLD_DISTANCE)
        collection[(idx_wf, idx_sd, idx1, idx2)] = (
            np.mean(scores),
            np.median(scores),
            np.std(scores),
            np.min(scores),
            np.max(scores)
        )

    with open(DATA_DIR / f"test_output_collection_{timestamp}.pickle", "wb") as f:
        pickle.dump(collection, f)


if __name__ == "__main__":
    main()
