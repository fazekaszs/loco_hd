from loco_hd import TagPairingRule, LoCoHD, PrimitiveAtom, WeightFunction

import unittest


class TestTagPairingRule(unittest.TestCase):

    def test_accept_same(self):
        """Tests tag pairing rules initialized with the `accept_same` key."""

        tpr = TagPairingRule({"accept_same": True})

        self.assertTrue(tpr.pair_accepted(("A", "A")))
        self.assertFalse(tpr.pair_accepted(("A", "B")))

        tpr = TagPairingRule({"accept_same": False})

        self.assertFalse(tpr.pair_accepted(("A", "A")))
        self.assertTrue(tpr.pair_accepted(("A", "B")))

    def test_tag_pairs(self):
        """Tests tag pairing rules initialized with the `tag_pairs`, `accepted_pairs` and `ordered` keys."""

        tpr = TagPairingRule({
            "tag_pairs": {("A", "B"), ("A", "C"), ("B", "C")},
            "accepted_pairs": True,
            "ordered": True
        })

        self.assertFalse(tpr.pair_accepted(("A", "A")))
        self.assertFalse(tpr.pair_accepted(("B", "B")))
        self.assertFalse(tpr.pair_accepted(("C", "C")))

        self.assertTrue(tpr.pair_accepted(("A", "B")))
        self.assertFalse(tpr.pair_accepted(("B", "A")))

        self.assertTrue(tpr.pair_accepted(("A", "C")))
        self.assertFalse(tpr.pair_accepted(("C", "A")))

        self.assertTrue(tpr.pair_accepted(("B", "C")))
        self.assertFalse(tpr.pair_accepted(("C", "B")))

        tpr = TagPairingRule({
            "tag_pairs": {("A", "B"), ("A", "C"), ("B", "C")},
            "accepted_pairs": True,
            "ordered": False
        })

        self.assertFalse(tpr.pair_accepted(("A", "A")))
        self.assertFalse(tpr.pair_accepted(("B", "B")))
        self.assertFalse(tpr.pair_accepted(("C", "C")))

        self.assertTrue(tpr.pair_accepted(("A", "B")))
        self.assertTrue(tpr.pair_accepted(("B", "A")))

        self.assertTrue(tpr.pair_accepted(("A", "C")))
        self.assertTrue(tpr.pair_accepted(("C", "A")))

        self.assertTrue(tpr.pair_accepted(("B", "C")))
        self.assertTrue(tpr.pair_accepted(("C", "B")))

        tpr = TagPairingRule({
            "tag_pairs": {("A", "B"), ("A", "C"), ("B", "C")},
            "accepted_pairs": False,
            "ordered": True
        })

        self.assertTrue(tpr.pair_accepted(("A", "A")))
        self.assertTrue(tpr.pair_accepted(("B", "B")))
        self.assertTrue(tpr.pair_accepted(("C", "C")))

        self.assertFalse(tpr.pair_accepted(("A", "B")))
        self.assertTrue(tpr.pair_accepted(("B", "A")))

        self.assertFalse(tpr.pair_accepted(("A", "C")))
        self.assertTrue(tpr.pair_accepted(("C", "A")))

        self.assertFalse(tpr.pair_accepted(("B", "C")))
        self.assertTrue(tpr.pair_accepted(("C", "B")))

        tpr = TagPairingRule({
            "tag_pairs": {("A", "B"), ("A", "C"), ("B", "C")},
            "accepted_pairs": False,
            "ordered": False
        })

        self.assertTrue(tpr.pair_accepted(("A", "A")))
        self.assertTrue(tpr.pair_accepted(("B", "B")))
        self.assertTrue(tpr.pair_accepted(("C", "C")))

        self.assertFalse(tpr.pair_accepted(("A", "B")))
        self.assertFalse(tpr.pair_accepted(("B", "A")))

        self.assertFalse(tpr.pair_accepted(("A", "C")))
        self.assertFalse(tpr.pair_accepted(("C", "A")))

        self.assertFalse(tpr.pair_accepted(("B", "C")))
        self.assertFalse(tpr.pair_accepted(("C", "B")))

    def test_in_locohd(self):
        """Test the behavior of tag pairing rules in small LoCoHD calculations."""

        # The z coordinate is not used, all primitive atoms are on the xy plane.
        primitive_structure = [
            # In "residue A" all primitive atoms are of type "A".
            PrimitiveAtom("A", "A", [0., 0., 0.]),
            PrimitiveAtom("A", "A", [0., 1., 0.]),
            PrimitiveAtom("A", "A", [2., 0., 0.]),
            PrimitiveAtom("A", "A", [2., 2., 0.]),
            # In "residue B" all primitive atoms are of type "B".
            PrimitiveAtom("B", "B", [1., 2., 0.]),
            PrimitiveAtom("B", "B", [1., 3., 0.]),
            PrimitiveAtom("B", "B", [3., 2., 0.]),
            PrimitiveAtom("B", "B", [3., 3., 0.]),
            # In "residue C" all primitive atoms are of type "C".
            PrimitiveAtom("C", "C", [2., 1., 0.]),
        ]

        # Since all "residues" have their own primitive types,
        #  "accept_same" = True will always yield a LoCoHD of 0 (for same residue pairings)
        #  or 1 (for different residue pairings).
        weight_function = WeightFunction("uniform", [1., 1.001])
        tpr = TagPairingRule({"accept_same": True})
        lchd = LoCoHD(["A", "B", "C"], weight_function, tpr)

        scores = lchd.from_primitives(
            primitive_structure,
            primitive_structure,
            [
                (0, 3),  # must be 0, same "residue"
                (4, 5),  # must be 0, same "residue"
                (0, 4),  # must be 1, different "residue"
                (0, 8),  # must be 1, different "residue"
                (4, 8),  # must be 1, different "residue"
            ],
            1.002
        )

        correct_scores = [0., 0., 1., 1., 1.]
        for score1, score2 in zip(scores, correct_scores):
            self.assertAlmostEqual(score1, score2, places=15)

        # Cases, when "accept_same" = False.
        weight_function = WeightFunction("uniform", [1., 1.001])
        tpr = TagPairingRule({"accept_same": False})
        lchd = LoCoHD(["A", "B", "C"], weight_function, tpr)

        scores = lchd.from_primitives(
            primitive_structure,
            primitive_structure,
            [(0, 3), (4, 5), (0, 4), (0, 8), (4, 8)],
            1.002
        )
        correct_scores = [0.7071, 0.5412, 0.5412, 0.4284, 0.6501]

        for score1, score2 in zip(scores, correct_scores):
            self.assertAlmostEqual(score1, score2, places=4)
