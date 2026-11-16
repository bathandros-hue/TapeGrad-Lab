"""Neural network layers and SGD written on top of Value."""
from __future__ import annotations

import random
from collections.abc import Sequence

from .engine import Value


class Neuron:
    def __init__(self, inputs: int, rng: random.Random):
        if inputs < 1:
            raise ValueError("Neuron requires at least one input")
        self.weights = [Value(rng.uniform(-1, 1)) for _ in range(inputs)]
        self.bias = Value(0)

    def __call__(self, values: Sequence[float | Value]) -> Value:
        if len(values) != len(self.weights):
            raise ValueError("Wrong input width")
        return (sum((w * x for w, x in zip(self.weights, values)), self.bias)).tanh()

    def parameters(self) -> list[Value]:
        return [*self.weights, self.bias]


class MLP:
    def __init__(self, dimensions: Sequence[int], seed: int = 42):
        if len(dimensions) < 2 or any(n < 1 for n in dimensions):
            raise ValueError("Need at least input and output dimensions, all positive")
        rng = random.Random(seed)
        self.layers = [
            [Neuron(dimensions[i], rng) for _ in range(dimensions[i + 1])]
            for i in range(len(dimensions) - 1)
        ]

    def __call__(self, xs: Sequence[float | Value]) -> list[Value]:
        out = list(xs)
        for layer in self.layers:
            out = [neuron(out) for neuron in layer]
        return out

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for neuron in layer for p in neuron.parameters()]


class SGD:
    def __init__(self, parameters: Sequence[Value], lr: float = 0.1, clip: float | None = None):
        if lr <= 0 or (clip is not None and clip <= 0):
            raise ValueError("Learning rate and optional clipping threshold must be positive")
        self.parameters = list(parameters)
        self.lr, self.clip = lr, clip

    def step(self) -> None:
        for param in self.parameters:
            grad = param.grad
            if self.clip is not None:
                grad = max(-self.clip, min(self.clip, grad))
            param.data -= self.lr * grad
