import random
from typing import List, Optional, Sequence, Tuple

from .board import BattleshipBoard


def _batch_random_board(
    args: Tuple["BattleshipBoard", Sequence[str], int],
) -> List[int]:
    """Generate *batch* boards — pickleable helper for parallel workers."""
    board, names, batch = args
    return board.random_board(names=names, batch_size=batch)


class BattleshipPlayer:
    """Monte-Carlo driven Battleship opponent.

    Maintains a pool of random boards consistent with the known game state
    (hits and misses).  On each turn the most frequent unseen tile across
    the sample is chosen as the next shot.

    Args:
        dim: Board side length.
        ships: List of ship lengths.
        boards: Number of Monte-Carlo samples to maintain.
        parallel_backend: ``'joblib'``, ``'multiprocessing'``, or anything
            else for sequential generation.
        n_jobs: Worker count for parallel backends.
    """

    def __init__(
        self,
        dim: int = 10,
        ships: Optional[Sequence[int]] = None,
        boards: int = 10000,
        parallel_backend: str = "sequential",
        n_jobs: int = 1,
    ) -> None:
        if ships is None:
            ships = [5, 4, 3, 3, 2]
        self.board = BattleshipBoard(dim=dim, ships=ships)

        enemy_bits = self.board.random_board(initial=True)[0]
        self.enemy_ships = [self.board.bit_to_coords(b) for b in enemy_bits]
        self.enemy_board = set.union(*self.enemy_ships)
        self.enemy_ship_statuses = {
            name: {"sunk": False, "confirmed_pos": 0, "hit_count": 0}
            for name in self.board.names
        }

        self.n_boards = boards
        self.parallel_backend = parallel_backend
        self.n_jobs = n_jobs

        self.hits: set = set()
        self.misses: set = set()
        self.random_boards: list = []
        self.hits_bit = 0
        self.misses_bit = 0
        self.name_order = list(self.board.names)
        self.target_hits = sum(self.board.ship_lengths.values())
        self.last_move = 0
        self.num_hits = 0

    # ------------------------------------------------------------------
    # Board sampling
    # ------------------------------------------------------------------

    def generate_random_boards(self) -> None:
        """Refresh the Monte-Carlo pool to respect current hits/misses.

        Previously valid boards are filtered first; only the deficit is
        newly generated, keeping sampling responsive at high *n_boards*.
        """
        if self.last_move == 1:
            self.random_boards = [
                b for b in self.random_boards if not (b & self.misses_bit)
            ]
        else:
            self.random_boards = [
                b
                for b in self.random_boards
                if (b & self.hits_bit) == self.hits_bit
            ]

        self.name_order = sorted(
            self.name_order,
            key=lambda x: self.board.poss_ships_num.get(x, float("inf")),
        )
        needed = self.n_boards - len(self.random_boards)
        if needed <= 0:
            return

        if self.parallel_backend == "joblib":
            from joblib import Parallel, delayed

            batch = (needed + self.n_jobs - 1) // self.n_jobs
            chunks: list[int] = []
            remain = needed
            for _ in range(self.n_jobs):
                if remain <= 0:
                    break
                cur = min(batch, remain)
                chunks.append(cur)
                remain -= cur

            results = Parallel(n_jobs=len(chunks), backend="loky")(
                delayed(_batch_random_board)((self.board, self.name_order, c))
                for c in chunks
            )
            for boards in results:
                self.random_boards.extend(boards)

        elif self.parallel_backend == "multiprocessing":
            import multiprocessing as mp

            batch = (needed + self.n_jobs - 1) // self.n_jobs
            chunks: list[int] = []  # type: ignore[no-redef]
            remain = needed
            for _ in range(self.n_jobs):
                if remain <= 0:
                    break
                cur = min(batch, remain)
                chunks.append(cur)
                remain -= cur

            with mp.Pool(len(chunks)) as pool:
                results = pool.map(
                    _batch_random_board,
                    [(self.board, self.name_order, c) for c in chunks],
                )
            for boards in results:
                self.random_boards.extend(boards)

        else:
            self.random_boards += self.board.random_board(
                names=self.name_order,
                batch_size=needed,
            )

    # ------------------------------------------------------------------
    # Ship position narrowing
    # ------------------------------------------------------------------

    def update_possible_ship_positions(self) -> None:
        """Prune impossible ship placements given current game state."""
        hits_bit = self.hits_bit
        misses_bit = self.misses_bit

        for name in self.board.names:
            if len(self.board.poss_ships[name]) == 1:
                continue
            necessary = self.enemy_ship_statuses[name]["confirmed_pos"]
            if necessary:
                need_len = self.enemy_ship_statuses[name]["hit_count"]
                if self.last_move == 1:
                    self.board.poss_ships[name] = [
                        conf
                        for conf in self.board.poss_ships[name]
                        if not (conf & misses_bit)
                    ]
                else:
                    self.board.poss_ships[name] = [
                        conf
                        for conf in self.board.poss_ships[name]
                        if (conf & necessary).bit_count() == need_len
                        and (conf & hits_bit).bit_count() == need_len
                    ]
            else:
                if self.last_move == 1:
                    self.board.poss_ships[name] = [
                        conf
                        for conf in self.board.poss_ships[name]
                        if not (conf & misses_bit)
                    ]
                else:
                    self.board.poss_ships[name] = [
                        conf
                        for conf in self.board.poss_ships[name]
                        if not (conf & hits_bit)
                    ]
            self.board.poss_ships_num[name] = len(self.board.poss_ships[name])

    # ------------------------------------------------------------------
    # Turn logic
    # ------------------------------------------------------------------

    def take_turn(self) -> Optional[Tuple[int, int]]:
        """Return the best tile to fire at based on Monte-Carlo frequency.

        Tallies tile frequencies across sampled boards and picks the most
        common unseen tile.  Falls back to a random unseen tile when the
        sample is exhausted.
        """
        self.generate_random_boards()
        dim2 = self.board.dim2
        counts = [0] * dim2
        for b in self.random_boards:
            bit = b
            while bit:
                lsb = bit & -bit
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                bit &= bit - 1

        unseen_mask = ~(self.hits_bit | self.misses_bit)
        best_idx = None
        best_count = -1
        for idx, c in enumerate(counts):
            if unseen_mask & (1 << idx) and c > best_count:
                best_idx = idx
                best_count = c

        if best_idx is not None:
            return divmod(best_idx, self.board.dim)

        remaining = [
            (x, y)
            for x in range(self.board.dim)
            for y in range(self.board.dim)
            if (x, y) not in self.hits and (x, y) not in self.misses
        ]
        if remaining:
            return random.choice(remaining)
        return None

    # ------------------------------------------------------------------
    # Game state updates
    # ------------------------------------------------------------------

    def update_game_state(self, x: int, y: int) -> None:
        """Record the outcome of firing at (x, y)."""
        coords = (x, y)
        bit = 1 << (x * self.board.dim + y)
        if coords in self.enemy_board:
            if coords not in self.hits:
                self.hits.add(coords)
                self.hits_bit |= bit
                self.num_hits += 1
                self.last_move = 2
                for name, ship in zip(self.board.names, self.enemy_ships):
                    if coords in ship:
                        self.enemy_ship_statuses[name]["hit_count"] += 1
                        self.enemy_ship_statuses[name]["confirmed_pos"] |= bit
                        if self.enemy_ship_statuses[name]["hit_count"] == len(
                            ship
                        ):
                            self.enemy_ship_statuses[name]["sunk"] = True
                        break
                self.update_possible_ship_positions()
        elif coords not in self.misses:
            self.last_move = 1
            self.misses.add(coords)
            self.misses_bit |= bit
            self.update_possible_ship_positions()

    def check_all_sunk(self) -> bool:
        """Return True when every enemy ship has been sunk."""
        return self.num_hits == self.target_hits

    def reset(self) -> None:
        """Reset all game state for a fresh round."""
        self.board.generate_component_layouts()
        enemy_bits = self.board.random_board(initial=True)[0]
        self.enemy_ships = [self.board.bit_to_coords(b) for b in enemy_bits]
        self.enemy_board = set.union(*self.enemy_ships)
        self.enemy_ship_statuses = {
            name: {"sunk": False, "confirmed_pos": 0, "hit_count": 0}
            for name in self.board.names
        }
        self.last_move = 0
        self.num_hits = 0
        self.hits, self.misses, self.random_boards = set(), set(), []
        self.hits_bit = 0
        self.misses_bit = 0
        self.name_order = list(self.board.names)
