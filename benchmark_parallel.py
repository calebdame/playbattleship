import time
from battleship import BattleshipPlayer


def benchmark(backend: str, jobs: int = 2, boards: int = 10000) -> float:
    """Return the time to generate ``boards`` boards using ``backend``."""
    player = BattleshipPlayer(dim=10, boards=boards,
                              parallel_backend=backend, n_jobs=jobs)
    start = time.perf_counter()
    player.generate_random_boards()
    return time.perf_counter() - start


if __name__ == "__main__":
    for backend in ["sequential", "joblib", "multiprocessing"]:
        duration = benchmark(backend=backend)
        print(f"{backend}: {duration:.4f}s")
