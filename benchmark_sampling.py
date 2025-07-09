import time

from battleship import BattleshipBoard


def benchmark(dim: int = 10, boards: int = 1000) -> float:
    """Return the time to generate ``boards`` random boards."""
    board = BattleshipBoard(dim=dim)
    start = time.perf_counter()
    board.random_board(batch_size=boards)
    return time.perf_counter() - start


if __name__ == "__main__":
    for n in [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]:
        duration = benchmark(boards=n)
        print(f"n={n}: {duration:.4f}s")
