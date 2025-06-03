from loco_hd import WeightFunction, LoCoHD, PrimitiveAtom

import os
import time
import numpy as np
import unittest
import pickle
from pathlib import Path


class TestLoCoHDValues(unittest.TestCase):

    def setUp(self):

        self.t_build_list = list()
        self.t_run_list = list()
        self.measure_times = False

    def tearDown(self):

        if not self.measure_times:
            return

        print(f"Total build time in test_consistency: {sum(self.t_build_list)} s")
        print(f"Total run time in test_consistency: {sum(self.t_run_list)} s")

    def test_small_locohd(self):
        """Small LoCoHD calculations using the from_anchors method and hand-calculated test-cases."""

        w_func = WeightFunction("uniform", [0., 4.])
        primitive_types = ["O", "A", "B", "C"]
        lchd = LoCoHD(primitive_types, w_func)

        sequence = ["O", "A", "B", "C"]

        value = lchd.from_anchors(sequence, sequence, [0., 1., 2., 3.], [0., 1., 1., 1.])
        self.assertAlmostEqual(value, 0.2268, places=4)

        value = lchd.from_anchors(sequence, sequence, [0., 1., 1., 1.], [0., 1., 2., 3.])
        self.assertAlmostEqual(value, 0.2268, places=4)

        w_func = WeightFunction("kumaraswamy", [3., 10., 2., 5.])
        primitive_types = ["A", "B", "C"]
        lchd = LoCoHD(primitive_types, w_func)

        seq_a = ["A", "B", "A", "C"]
        dists_a = [0., 1., 5., 9.]
        seq_b = ["A", "C"]
        dists_b = [0., 7.]

        value = lchd.from_anchors(seq_a, seq_b, dists_a, dists_b)
        self.assertAlmostEqual(value, 0.4979, places=4)

    def test_locohd_err(self):
        """Plugging invalid parameters into the LoCoHD constructor."""

        w_func = WeightFunction("uniform", [0., 4.])
        primitive_types = ["O", "A", "B", "C"]

        with self.assertRaises(ValueError):
            LoCoHD([], w_func)

        with self.assertRaises(ValueError):
            LoCoHD(primitive_types, w_func, category_weights=[1., 1., 1.])
        
        with self.assertRaises(ValueError):
            LoCoHD(primitive_types, w_func, category_weights=[1., 1., 1., 1., 1.])
        
        with self.assertRaises(ValueError):
            LoCoHD(primitive_types, w_func, category_weights=[1., -1., 1., 1.])
        
        with self.assertRaises(ValueError):
            LoCoHD(primitive_types, w_func, category_weights=[1., 0., 1., 1.])

    @unittest.skipIf(os.getenv("CONSISTENCY") is None, reason="CONSISTENCY test is not requested.")
    def test_consistency(self):
        """Large LoCoHD calculations using the from_primitives method and auto-generated test-cases."""

        self.measure_times = True
        data_dir = Path("tests/test_data")
        datafile_date = "241212_131752"

        with open(data_dir / f"test_input_collection_{datafile_date}.pickle", "rb") as f:
            input_collection = pickle.load(f)

        with open(data_dir / f"test_output_collection_{datafile_date}.pickle", "rb") as f:
            output_collection = pickle.load(f)

        for idx_wf, idx1, idx2 in output_collection:

            scp1 = input_collection["sequence_coordinate_pairs"][idx1]
            scp2 = input_collection["sequence_coordinate_pairs"][idx2]
            pras1 = [PrimitiveAtom(x, "", y) for x, y in zip(scp1["seq"], scp1["coords"])]
            pras2 = [PrimitiveAtom(x, "", y) for x, y in zip(scp2["seq"], scp2["coords"])]
            anchors = [(x, x) for x in range(min(len(pras1), len(pras2)))]

            w_func = WeightFunction(*input_collection["weight_functions"][idx_wf])

            t_build = time.time()
            lchd = LoCoHD(input_collection["primitive_types"], w_func)
            t_build = time.time() - t_build
            self.t_build_list.append(t_build)

            t_run = time.time()
            scores = lchd.from_primitives(pras1, pras2, anchors, input_collection["threshold_distance"])
            t_run = time.time() - t_run
            self.t_run_list.append(t_run)

            current_results = (
                np.mean(scores),
                np.median(scores),
                np.std(scores),
                np.min(scores),
                np.max(scores)                
            )

            target_results = output_collection[(idx_wf, idx1, idx2)]

            for target, current in zip(target_results, current_results):
                self.assertAlmostEqual(target, current, places=15)
