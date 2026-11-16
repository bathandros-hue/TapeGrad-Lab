# TapeGrad Lab

Tiny, inspectable reverse-mode autodiff with numerical gradient checks, an MLP, and an XOR training demo.

**Why it exists:** Inspect each derivative and compare automatic gradients with central finite differences; observe how backpropagation behaves on shared and deep computation graphs.

## Quickstart

From this repository root, using Python 3.10+:

    python -m unittest discover -s tests -v
    python examples/xor.py

No runtime dependencies or GPU required.

## Design

- Scalar Value objects capture each arithmetic operation in a dynamic computation graph.
- Iterative topological traversal computes reverse-mode derivatives without Python recursion limits.
- Backward clears graph gradients by default, avoiding accidental over-accumulation between steps.
- Gradient-checking compares analytic derivatives with central finite differences.
- Small MLP and clipped SGD demonstrate actual trainable behavior.

## Limitations

Educational scalar engine, not efficient for realistic neural networks. No batching, GPU backend, or general tensor autodiff. Check the example training loss for a reproducible XOR demonstration.

## Suggested GitHub topics

autograd, backpropagation, machine-learning, neural-networks, python, gradient-checking

## License

MIT; see LICENSE.
