# Battleship Monte-Carlo Engine

A Python module that plays Battleship using Monte-Carlo sampling. Designed to be cloned and imported as the AI backend for an interactive Battleship web app.

## Structure

```
battleship/
    __init__.py        # Exports BattleshipBoard, BattleshipPlayer
    board.py           # Board geometry, legal placements, random board generation
    player.py          # Monte-Carlo AI player (sampling, turn selection, state tracking)
    _fast_board.pyx    # Cython extension for fast board generation (~8-11x speedup)
tests/
    test_battleship.py   # Core logic tests
    test_fast_board.py   # Cython extension tests + benchmark
    test_parallel.py     # Parallel backend tests
setup.py               # Build script for Cython extension
```

## Installation

```bash
git clone https://github.com/calebdame/playbattleship.git
cd playbattleship
pip install -r requirements.txt
python setup.py build_ext --inplace   # optional: build Cython extension for ~8-11x faster board gen
```

## Usage

```python
from battleship import BattleshipPlayer

# Standard 10x10 board with default ships [5, 4, 3, 3, 2]
player = BattleshipPlayer(dim=10, boards=10000)

# Get the AI's best guess
coord = player.take_turn()        # e.g. (3, 7)

# Tell the AI the result (it checks against its internal enemy board)
player.update_game_state(*coord)

# Check for victory
if player.check_all_sunk():
    print("All ships sunk!")

# Start a new game
player.reset()
```

### Time-based sampling

Instead of a fixed sample count, you can give each turn a time budget.
The generation phase stops after `board_time - tol` seconds, so the turn
completes in roughly `board_time` seconds regardless of hardware speed.

```python
# Sample boards for ~3 seconds per turn (tol defaults to 1e-6)
player = BattleshipPlayer(dim=10, board_time=3.0)

# Tighten the early-stop tolerance if needed
player = BattleshipPlayer(dim=10, board_time=3.0, tol=1e-4)
```

After every call to `take_turn()` the timing and board count are recorded
in `player.turn_data`, a dict mapping turn number to `(time_took, n_boards_sampled)`:

```python
coord = player.take_turn()
player.update_game_state(*coord)

turn_time, n_boards = player.turn_data[1]
print(f"Turn 1: {n_boards} boards sampled in {turn_time:.3f}s")
```

`turn_data` is populated in both count-based and time-based modes.
It is cleared by `player.reset()`.

### Using as a backend for your own game

When integrating with your own server, you typically manage the enemy board yourself. The key methods:

| Method | Purpose |
|---|---|
| `BattleshipPlayer(dim, ships, boards)` | Create a player with a fixed Monte-Carlo sample size |
| `BattleshipPlayer(dim, ships, board_time, tol)` | Create a player with a per-turn time budget |
| `player.take_turn()` | Returns `(row, col)` — the AI's best guess |
| `player.update_game_state(row, col)` | Records hit/miss and updates internal sampling |
| `player.check_all_sunk()` | Returns `True` when all ships are sunk |
| `player.reset()` | Clears state for a new game |
| `player.turn_data` | `{turn_number: (time_took, n_boards_sampled)}` for every turn |
| `player.generate_random_boards()` | Manually refresh the Monte-Carlo pool |

### Parallel board generation

For higher sample counts, use joblib or multiprocessing:

```python
player = BattleshipPlayer(
    boards=50000,
    parallel_backend="joblib",  # or "multiprocessing"
    n_jobs=4,
)
```

### Cython acceleration

The Cython extension (`_fast_board.pyx`) replaces the rejection-sampling loop
with typed C using 128-bit bitboards and a xorshift64* PRNG. It supports boards
up to 11x11. If the extension is not built, `board.py` falls back to pure Python
automatically.

## How it works

1. **Pre-computation** — All legal ship placements are encoded as bitboard integers during initialization.
2. **Sampling** — Each turn, a pool of random boards is built. In count-based mode (`boards=N`), previously valid boards are filtered and only the deficit is regenerated. In time-based mode (`board_time=T`), boards are generated fresh for `T - tol` seconds using break statements in both the Python and Cython inner loops.
3. **Selection** — Tile frequencies across the sample are tallied. The most common unseen tile is chosen as the next shot.
4. **Pruning** — After each shot, impossible ship placements are removed, accelerating future sampling.
5. **Telemetry** — Every turn records `(elapsed_seconds, boards_sampled)` in `player.turn_data` keyed by turn number.

## Running tests

```bash
pip install pytest
pytest tests/
```
