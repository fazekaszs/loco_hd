import numpy as np
import itertools
import pickle
from datetime import datetime
from pathlib import Path

from loco_hd import WeightFunction, LoCoHD, PrimitiveAtom


PRIMITIVE_TYPES = [
    "A", "B", "C", "D", "E"
]
THRESHOLD_DISTANCE = 50.
DATA_DIR = Path("tests/test_data")


def generate_wf_params():

    # Generate random weight functions

    weight_functions = list()

    # hyper_exp, two parameter type
    for _ in range(5):
        param_b = 1. / np.random.uniform(3., 20.)
        weight_functions.append((
            "hyper_exp", [1., param_b]
        ))
    
    # hyper_exp, six parameter type
    for _ in range(5):
        param_a = np.random.uniform(1E-5, 1., size=3)
        param_b = 1. / np.random.uniform(3., 20., size=3)
        weight_functions.append((
            "hyper_exp", np.concatenate([param_a, param_b]).tolist()
        ))
    
    # dagum
    for _ in range(10):
        param_a = np.random.uniform(0.5, 3.0)
        param_b = np.random.uniform(1.5, 4.0)
        param_c = np.random.uniform(1., 25.)
        weight_functions.append((
            "dagum", [param_a, param_b, param_c]
        ))
    
    # uniform
    for _ in range(10):
        param_a = np.random.uniform(0.5, 10.0)
        delta_ab = np.random.uniform(0.1, 10.0)
        weight_functions.append((
            "uniform", [param_a, param_a + delta_ab]
        ))
    
    # kumaraswamy
    for _ in range(10):
        param_a = np.random.uniform(0.5, 10.0)
        delta_ab = np.random.uniform(0.1, 10.0)
        param_c = np.random.uniform(1. + 1E-5, 10.)
        param_d = np.random.uniform(1. + 1E-5, 10.)
        weight_functions.append((
            "kumaraswamy", [param_a, param_a + delta_ab, param_c, param_d]
        ))

    return weight_functions


def generate_sequence_coord_pairs():

    seq_coord_pairs = list()

    for _ in range(40):

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
    
    wfs = generate_wf_params()
    scps = generate_sequence_coord_pairs()

    with open(DATA_DIR / f"test_input_collection_{timestamp}.pickle", "wb") as f:
        pickle.dump({
            "primitive_types": PRIMITIVE_TYPES,
            "threshold_distance": THRESHOLD_DISTANCE,
            "weight_functions": wfs,
            "sequence_coordinate_pairs": scps
        }, f)

    iterator = itertools.product(
        enumerate(wfs), 
        enumerate(scps), 
        enumerate(scps)
    )

    collection = dict()
    
    for (idx_wf, wf), (idx1, scp1), (idx2, scp2) in iterator:

        if idx1 == idx2:
            continue
        
        lchd = LoCoHD(PRIMITIVE_TYPES, WeightFunction(*wf))
        pras1 = [PrimitiveAtom(x, "", y) for x, y in zip(scp1["seq"], scp1["coords"])]
        pras2 = [PrimitiveAtom(x, "", y) for x, y in zip(scp2["seq"], scp2["coords"])]
        anchors = [(x, x) for x in range(min(len(pras1), len(pras2)))]
        scores = lchd.from_primitives(pras1, pras2, anchors, THRESHOLD_DISTANCE)
        collection[(idx_wf, idx1, idx2)] = (
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
