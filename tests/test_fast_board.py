"""Tests for the Cython _fast_board extension.

These mirror the core board-generation tests in test_battleship.py but
exercise the C path directly (and also via BattleshipBoard.random_board
which delegates automatically when the extension is available).
"""

import random
import time

import pytest

from battleship import BattleshipBoard, BattleshipPlayer

try:
    from battleship._fast_board import fast_random_boards
except ImportError:
    fast_random_boards = None

pytestmark = pytest.mark.skipif(
    fast_random_boards is None,
    reason="Cython extension not built",
)


@pytest.fixture(autouse=True)
def fixed_seed():
    random.seed(42)


# ── Direct fast_random_boards tests ──────────────────────────────


def test_fast_gen_basic_combined():
    """Combined bitboard: correct count, no overlaps."""
    board = BattleshipBoard(dim=4, ships=[2, 3])
    placements = [board.poss_ships[n] for n in board.names]
    counts = [board.poss_ships_num[n] for n in board.names]
    results = fast_random_boards(placements, counts, 50, False)
    assert len(results) == 50
    for b in results:
        coords = board.bit_to_coords(b)
        assert len(coords) == 5  # ship lengths 2 + 3


def test_fast_gen_initial_per_ship():
    """initial=True returns list-of-lists with per-ship bitboards."""
    board = BattleshipBoard(dim=4, ships=[2, 2])
    placements = [board.poss_ships[n] for n in board.names]
    counts = [board.poss_ships_num[n] for n in board.names]
    results = fast_random_boards(placements, counts, 30, True)
    assert len(results) == 30
    for ships in results:
        assert isinstance(ships, list)
        assert len(ships) == 2
        combined = 0
        for s in ships:
            assert s & combined == 0, "ships overlap"
            combined |= s


def test_fast_gen_single_ship():
    """Works with a single ship."""
    board = BattleshipBoard(dim=4, ships=[2])
    placements = [board.poss_ships["a"]]
    counts = [board.poss_ships_num["a"]]
    results = fast_random_boards(placements, counts, 20, False)
    assert len(results) == 20
    for b in results:
        assert bin(b).count("1") == 2


def test_fast_gen_empty():
    """Edge cases: no ships or zero batch."""
    assert fast_random_boards([], [], 10, False) == []
    board = BattleshipBoard(dim=4, ships=[2])
    placements = [board.poss_ships["a"]]
    counts = [board.poss_ships_num["a"]]
    assert fast_random_boards(placements, counts, 0, False) == []


# ── Integration via BattleshipBoard.random_board ─────────────────


def test_random_board_uses_fast_path():
    """BattleshipBoard.random_board delegates to Cython for dim<=11."""
    board = BattleshipBoard(dim=10, ships=[5, 4, 3, 3, 2])
    results = board.random_board(batch_size=100)
    assert len(results) == 100
    total_tiles = sum(board.ship_lengths.values())
    for b in results:
        assert bin(b).count("1") == total_tiles


def test_random_board_initial_via_fast():
    board = BattleshipBoard(dim=10, ships=[5, 4, 3, 3, 2])
    results = board.random_board(initial=True, batch_size=5)
    assert len(results) == 5
    for ships in results:
        assert len(ships) == 5
        combined = 0
        for s in ships:
            assert s & combined == 0
            combined |= s


# ── Full game still works with Cython ────────────────────────────


def test_full_game_with_cython():
    """End-to-end game on 5x5 board to verify Cython integration."""
    player = BattleshipPlayer(dim=5, ships=[3, 2], boards=1000)
    turns = 0
    while not player.check_all_sunk():
        coord = player.take_turn()
        assert coord is not None
        player.update_game_state(*coord)
        turns += 1
        assert turns <= 25
    assert player.check_all_sunk()


# ── Benchmark (not strict, just for visibility) ──────────────────


def test_benchmark_cython_vs_python(capsys):
    """Print timing comparison so you can see the speedup."""
    board = BattleshipBoard(dim=10, ships=[5, 4, 3, 3, 2])
    n = 10000

    # Cython path
    random.seed(0)
    t0 = time.perf_counter()
    board.random_board(batch_size=n)
    t_cython = time.perf_counter() - t0

    # Force Python fallback
    import battleship.board as _bmod
    orig = _bmod._fast_gen
    _bmod._fast_gen = None
    try:
        random.seed(0)
        t0 = time.perf_counter()
        board.random_board(batch_size=n)
        t_python = time.perf_counter() - t0
    finally:
        _bmod._fast_gen = orig

    with capsys.disabled():
        print(
            f"\n  {n} boards: "
            f"Cython {t_cython:.3f}s, "
            f"Python {t_python:.3f}s, "
            f"speedup {t_python / t_cython:.1f}x"
        )
    # Just assert Cython isn't slower
    assert t_cython <= t_python * 1.5
