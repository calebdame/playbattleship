# Battleship Monte-Carlo Engine

A Python module that plays Battleship using Monte-Carlo sampling. Designed to be cloned and imported as the AI backend for an interactive Battleship web app.

## Structure

```
battleship/
    __init__.py      # Exports BattleshipBoard, BattleshipPlayer
    board.py         # Board geometry, legal placements, random board generation
    player.py        # Monte-Carlo AI player (sampling, turn selection, state tracking)
tests/
    test_battleship.py   # Core logic tests
    test_parallel.py     # Parallel backend tests
```

## Installation

```bash
git clone https://github.com/calebdame/playbattleship.git
cd playbattleship
pip install -r requirements.txt
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

### Using as a backend for your own game

When integrating with your own server, you typically manage the enemy board yourself. The key methods:

| Method | Purpose |
|---|---|
| `BattleshipPlayer(dim, ships, boards)` | Create a player with given board config and sample size |
| `player.take_turn()` | Returns `(row, col)` — the AI's best guess |
| `player.update_game_state(row, col)` | Records hit/miss and updates internal sampling |
| `player.check_all_sunk()` | Returns `True` when all ships are sunk |
| `player.reset()` | Clears state for a new game |
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

## How it works

1. **Pre-computation** — All legal ship placements are encoded as bitboard integers during initialization.
2. **Sampling** — Each turn, a pool of random boards consistent with known hits/misses is maintained. Previously valid boards are filtered first; only the deficit is newly generated.
3. **Selection** — Tile frequencies across the sample are tallied. The most common unseen tile is chosen as the next shot.
4. **Pruning** — After each shot, impossible ship placements are removed, accelerating future sampling.

## Running tests

```bash
pip install pytest
pytest tests/
```
