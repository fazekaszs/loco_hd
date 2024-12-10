from loco_hd import WeightFunction

import unittest


class TestWeightFunctions(unittest.TestCase):

    def test_hyper_exp_ok(self):

        wf = WeightFunction("hyper_exp", [1., 1.])

        self.assertAlmostEqual(wf.integral_range(0., 1.), 0.6321, places=4)
        self.assertAlmostEqual(wf.integral_range(1., 3.), 0.3181, places=4)
        self.assertAlmostEqual(wf.integral_range(5., 10.), 0.0067, places=4)

        wf = WeightFunction("hyper_exp", [0.5, 0.5, 1. / 2., 1. / 3.])

        self.assertAlmostEqual(wf.integral_range(0., 1.), 0.3385, places=4)
        self.assertAlmostEqual(wf.integral_range(1., 3.), 0.3660, places=4)
        self.assertAlmostEqual(wf.integral_range(5., 10.), 0.1143, places=4)

        wf = WeightFunction("hyper_exp", [3., 5., 2., 1. / 3., 1. / 5., 1. / 10.])

        self.assertAlmostEqual(wf.integral_range(0., 1.), 0.1947, places=4)
        self.assertAlmostEqual(wf.integral_range(1., 3.), 0.2724, places=4)
        self.assertAlmostEqual(wf.integral_range(5., 10.), 0.2100, places=4)

    def test_hyper_exp_err(self):

        with self.assertRaises(ValueError):
            WeightFunction("hyper_exp", [1., ])

        with self.assertRaises(ValueError):
            WeightFunction("hyper_exp", [1., 2., 3.])

        with self.assertRaises(ValueError):
            WeightFunction("hyper_exp", [-1., 1.])

        with self.assertRaises(ValueError):
            WeightFunction("hyper_exp", [1., -1.])

        with self.assertRaises(ValueError):
            WeightFunction("hyper_exp", [1., -1., 2.])

    def test_dagum_ok(self):

        wf = WeightFunction("dagum", [1., 1., 1.])

        self.assertAlmostEqual(wf.integral_range(0., 1.), 0.5000, places=4)
        self.assertAlmostEqual(wf.integral_range(1., 3.), 0.2500, places=4)
        self.assertAlmostEqual(wf.integral_range(5., 10.), 0.0758, places=4)

        wf = WeightFunction("dagum", [2., 5., 1.])

        self.assertAlmostEqual(wf.integral_range(0., 1.), 0.0385, places=4)
        self.assertAlmostEqual(wf.integral_range(1., 3.), 0.2262, places=4)
        self.assertAlmostEqual(wf.integral_range(5., 10.), 0.3000, places=4)

        wf = WeightFunction("dagum", [10., 5., 2.])

        self.assertAlmostEqual(wf.integral_range(0., 1.), 0.0000, places=4)
        self.assertAlmostEqual(wf.integral_range(1., 3.), 0.0000, places=4)
        self.assertAlmostEqual(wf.integral_range(5., 10.), 0.7480, places=4)

    def test_dagum_err(self):

        with self.assertRaises(ValueError):
            WeightFunction("dagum", [1., ])

        with self.assertRaises(ValueError):
            WeightFunction("dagum", [1., 2.])

        with self.assertRaises(ValueError):
            WeightFunction("dagum", [-1., 2., 3.])

        with self.assertRaises(ValueError):
            WeightFunction("dagum", [1., -2., 3.])

        with self.assertRaises(ValueError):
            WeightFunction("dagum", [1., 2., -3.])

    def test_uniform_ok(self):

        wf = WeightFunction("uniform", [0., 1.])

        self.assertAlmostEqual(wf.integral_range(0., 1.), 1.0000, places=4)
        self.assertAlmostEqual(wf.integral_range(1., 3.), 0.0000, places=4)
        self.assertAlmostEqual(wf.integral_range(5., 10.), 0.0000, places=4)

        wf = WeightFunction("uniform", [3., 10.])

        self.assertAlmostEqual(wf.integral_range(0., 1.), 0.0000, places=4)
        self.assertAlmostEqual(wf.integral_range(1., 3.), 0.0000, places=4)
        self.assertAlmostEqual(wf.integral_range(5., 10.), 0.7143, places=4)
        
        wf = WeightFunction("uniform", [2., 16.])

        self.assertAlmostEqual(wf.integral_range(0., 1.), 0.0000, places=4)
        self.assertAlmostEqual(wf.integral_range(1., 3.), 0.0714, places=4)
        self.assertAlmostEqual(wf.integral_range(5., 10.), 0.3571, places=4)        

    def test_uniform_err(self):

        with self.assertRaises(ValueError):
            WeightFunction("uniform", [1., ])

        with self.assertRaises(ValueError):
            WeightFunction("uniform", [1., 0.])

        with self.assertRaises(ValueError):
            WeightFunction("uniform", [-1., 0.])

    def test_kumaraswamy_ok(self):

        wf = WeightFunction("kumaraswamy", [1., 2., 2., 2.])

        self.assertAlmostEqual(wf.integral_range(1.0, 2.0), 1.0000, places=4)
        self.assertAlmostEqual(wf.integral_range(1.25, 1.75), 0.6875, places=4)
        self.assertAlmostEqual(wf.integral_range(1.4, 10.0), 0.7056, places=4)

        wf = WeightFunction("kumaraswamy", [5., 10., 2., 3.])

        self.assertAlmostEqual(wf.integral_range(5., 7.), 0.4073, places=4)
        self.assertAlmostEqual(wf.integral_range(1., 17.), 1.0000, places=4)
        self.assertAlmostEqual(wf.integral_range(6.4, 6.7), 0.0910, places=4)

        wf = WeightFunction("kumaraswamy", [5., 9., 7., 7.])

        self.assertAlmostEqual(wf.integral_range(5., 7.), 0.0534, places=4)
        self.assertAlmostEqual(wf.integral_range(1., 17.), 1.0000, places=4)
        self.assertAlmostEqual(wf.integral_range(6.4, 6.7), 0.0129, places=4)


if __name__ == "__main__":
    unittest.main()
