"""A minimal scalar reverse-mode autodiff engine with explicit graph traversal."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence


class Value:
    def __init__(self, data: float, children: tuple["Value", ...] = (), op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = children
        self._op = op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    @staticmethod
    def _value(other: float | "Value") -> "Value":
        return other if isinstance(other, Value) else Value(float(other))

    def __add__(self, other: float | "Value") -> "Value":
        other = self._value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    __radd__ = __add__

    def __mul__(self, other: float | "Value") -> "Value":
        other = self._value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    __rmul__ = __mul__

    def __pow__(self, power: int | float) -> "Value":
        if not isinstance(power, (int, float)):
            return NotImplemented
        out = Value(self.data**power, (self,), f"pow({power})")

        def backward() -> None:
            self.grad += power * (self.data ** (power - 1)) * out.grad if power else 0.0

        out._backward = backward
        return out

    def __neg__(self) -> "Value":
        return -1 * self

    def __sub__(self, other: float | "Value") -> "Value":
        return self + (-self._value(other))

    def __rsub__(self, other: float | "Value") -> "Value":
        return self._value(other) - self

    def __truediv__(self, other: float | "Value") -> "Value":
        return self * self._value(other) ** -1

    def __rtruediv__(self, other: float | "Value") -> "Value":
        return self._value(other) * self**-1

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def backward() -> None:
            self.grad += (1 - t * t) * out.grad

        out._backward = backward
        return out

    def exp(self) -> "Value":
        out = Value(math.exp(self.data), (self,), "exp")

        def backward() -> None:
            self.grad += out.data * out.grad

        out._backward = backward
        return out

    def log(self) -> "Value":
        out = Value(math.log(self.data), (self,), "log")

        def backward() -> None:
            self.grad += out.grad / self.data

        out._backward = backward
        return out

    def backward(self, seed: float = 1.0, reset: bool = True) -> None:
        """Differentiate through the graph iteratively, including shared nodes."""
        ordered: list[Value] = []
        seen: set[Value] = set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                ordered.append(node)
            elif node not in seen:
                seen.add(node)
                stack.append((node, True))
                stack.extend((parent, False) for parent in node._prev)
        if reset:
            for node in ordered:
                node.grad = 0.0
        self.grad += seed
        for node in reversed(ordered):
            node._backward()


def check_gradient(
    fn: Callable[[Sequence[Value]], Value],
    parameters: Sequence[float],
    epsilon: float = 1e-6,
) -> list[tuple[float, float]]:
    """Return (analytic, central finite-difference) derivative pairs."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    xs = [Value(x) for x in parameters]
    fn(xs).backward()
    result: list[tuple[float, float]] = []
    for i, analytic in enumerate(x.grad for x in xs):
        plus, minus = list(parameters), list(parameters)
        plus[i] += epsilon
        minus[i] -= epsilon
        fp = fn([Value(x) for x in plus]).data
        fm = fn([Value(x) for x in minus]).data
        result.append((analytic, (fp - fm) / (2 * epsilon)))
    return result
