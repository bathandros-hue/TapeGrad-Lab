import math
import unittest

from tapegrad import MLP, SGD, Value, check_gradient
from examples.xor import train


class TestAutodiff(unittest.TestCase):
    def test_shared_node(self):
        x = Value(3.0)
        y = x * x + 2 * x
        y.backward()
        self.assertAlmostEqual(x.grad, 8)
        y.backward()
        self.assertAlmostEqual(x.grad, 8)

    def test_mixed_gradients(self):
        grads = check_gradient(
            lambda xs: ((xs[0] * xs[1]).tanh() + xs[0].exp() + (xs[1] + 3).log()),
            [0.2, 0.7],
        )
        for analytic, numerical in grads:
            self.assertTrue(math.isclose(analytic, numerical, rel_tol=1e-5))

    def test_deep_graph_without_recursion(self):
        x = Value(1)
        result = x
        for _ in range(3000):
            result = result + 1
        result.backward()
        self.assertEqual(x.grad, 1)

    def test_validation_and_optimizer(self):
        with self.assertRaises(ValueError):
            MLP([2, 0])
        x = Value(1)
        x.grad = 100
        SGD([x], lr=.1, clip=2).step()
        self.assertAlmostEqual(x.data, 0.8)

    def test_xor_training(self):
        model, loss = train(steps=1200)
        self.assertLess(loss, 0.04)
        expected = { (0, 0): -1, (0, 1): 1, (1, 0): 1, (1, 1): -1 }
        for xs, label in expected.items():
            self.assertGreater(model(xs)[0].data * label, 0.7)


if __name__ == "__main__":
    unittest.main()
