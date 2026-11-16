"""Train a small MLP on XOR without tensor libraries."""
from tapegrad import MLP, SGD, Value


def train(steps: int = 1200, lr: float = 0.1) -> tuple[MLP, float]:
    points = [([0.0, 0.0], -1.0), ([0.0, 1.0], 1.0),
              ([1.0, 0.0], 1.0), ([1.0, 1.0], -1.0)]
    model = MLP([2, 6, 1], seed=11)
    optimizer = SGD(model.parameters(), lr=lr, clip=2.0)
    loss = Value(0.0)
    for _ in range(steps):
        loss = sum((model(x)[0] - label) ** 2 for x, label in points) / len(points)
        loss.backward()
        optimizer.step()
    return model, loss.data


if __name__ == "__main__":
    network, final_loss = train()
    print(f"XOR training loss (before final update): {final_loss:.6f}")
    for point in ([0, 0], [0, 1], [1, 0], [1, 1]):
        print(f"{point} -> {network(point)[0].data:+.4f}")
