from loco_hd import WeightFunction, LoCoHD, TagPairingRule

import unittest


class TestWeightFunctions(unittest.TestCase):

    def test_small_locohd(self):

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
