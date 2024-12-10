from loco_hd import WeightFunction, LoCoHD, TagPairingRule

import unittest


class TestWeightFunctions(unittest.TestCase):

    def test_small_locohd(self):

        w_func = WeightFunction("uniform", [0., 4.])
        primitive_types = ["O", "A", "B", "C"]
        sequence = ["O", "A", "B", "C"]
        lchd = LoCoHD(primitive_types, w_func)

        value = lchd.from_anchors(sequence, sequence, [0., 1., 2., 3.], [0., 1., 1., 1.])
        self.assertAlmostEqual(value, 0.2268, places=4)

        value = lchd.from_anchors(sequence, sequence, [0., 1., 1., 1.], [0., 1., 2., 3.])
        self.assertAlmostEqual(value, 0.2268, places=4)
