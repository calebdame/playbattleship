import time
from battleship import BattleshipBoard


def benchmark(dim=10, boards=1000):
    """Return the time to generate ``boards`` random boards."""
    b = BattleshipBoard(dim=dim)
    start = time.perf_counter()
    b.randomBoard(batch_size=boards)
    return time.perf_counter() - start


if __name__ == "__main__":
    duration = benchmark()
    print(f"Python sets: {duration:.4f}s")
